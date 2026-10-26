using System.Globalization;
using OpenCvSharp;
using OpenCvSharp.Aruco;

// Finds ArUco markers (dictionary 4x4_50) and prints the probe position inside the table square.
//
//   IDs 0-3 = corner markers, placed counter-clockwise as the camera sees them:
//       ID 3 ---- ID 2
//        |         |
//       ID 0 ---- ID 1          (ID 0 = table origin, x towards ID 1, y towards ID 3)
//   ID 4    = marker on top of the probe
//
// Usage:
//   dotnet run -- --size 300              camera 0, distance between marker CENTRES = 300 mm
//   dotnet run -- --size 300 --cam 1      use /dev/video1
//   dotnet run -- --size 300 --image photo.jpg    test on a photo instead of a camera
//
// While running with a camera, snapshot.jpg is saved about once per second (open it to check detection).

string? ArgValue(string name)
{
    int i = Array.IndexOf(args, name);
    return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
}

double side = double.Parse(ArgValue("--size") ?? "300", CultureInfo.InvariantCulture);
int camIndex = int.Parse(ArgValue("--cam") ?? "0");
string? imagePath = ArgValue("--image");
const int ProbeId = 4;

// Corner marker centres in table coordinates (mm)
var table = new Dictionary<int, Point2d>
{
    [0] = new Point2d(0, 0),
    [1] = new Point2d(side, 0),
    [2] = new Point2d(side, side),
    [3] = new Point2d(0, side),
};

using var dictionary = CvAruco.GetPredefinedDictionary(PredefinedDictionaryName.Dict4X4_50);
using var detector = new ArucoDetector(dictionary);

Mat? homography = null;                  // image pixels -> table mm
var printTimer = System.Diagnostics.Stopwatch.StartNew();

void Process(Mat frame)
{
    detector.DetectMarkers(frame, out var corners, out var ids, out _);

    var centers = new Dictionary<int, Point2d>();
    for (int i = 0; i < ids.Length; i++)
    {
        double cx = corners[i].Average(p => p.X);
        double cy = corners[i].Average(p => p.Y);
        centers[ids[i]] = new Point2d(cx, cy);

        var pt = new Point((int)cx, (int)cy);
        Cv2.Circle(frame, pt, 8, ids[i] == ProbeId ? Scalar.Red : Scalar.Lime, 2);
        Cv2.PutText(frame, $"ID {ids[i]}", new Point(pt.X + 10, pt.Y - 10),
                    HersheyFonts.HersheySimplex, 0.7, Scalar.Yellow, 2);
    }

    // Update the pixel->table mapping whenever all four corner markers are visible.
    if (table.Keys.All(centers.ContainsKey))
    {
        homography?.Dispose();
        homography = Cv2.FindHomography(table.Keys.Select(k => centers[k]),
                                        table.Keys.Select(k => table[k]));
    }

    string status;
    if (homography == null)
    {
        var missing = table.Keys.Where(k => !centers.ContainsKey(k));
        status = $"waiting for corner markers, missing ID: {string.Join(", ", missing)}";
    }
    else if (!centers.TryGetValue(ProbeId, out var probe))
    {
        status = "probe marker (ID 4) not visible";
    }
    else
    {
        var p = Cv2.PerspectiveTransform(new[] { probe }, homography)[0];
        bool inside = p.X >= 0 && p.X <= side && p.Y >= 0 && p.Y <= side;
        status = $"probe:  x = {p.X,7:F1} mm   y = {p.Y,7:F1} mm" + (inside ? "" : "   (outside the square)");
        Cv2.PutText(frame, $"x={p.X:F0} y={p.Y:F0} mm", new Point(10, 30),
                    HersheyFonts.HersheySimplex, 0.9, Scalar.Red, 2);
    }

    if (printTimer.ElapsedMilliseconds > 200)
    {
        printTimer.Restart();
        Console.WriteLine(status);
    }
}

// --- test on a still image -------------------------------------------------
if (imagePath != null)
{
    using var img = Cv2.ImRead(imagePath);
    if (img.Empty()) { Console.WriteLine($"Could not read image: {imagePath}"); return; }
    printTimer.Restart();
    Thread.Sleep(250);                   // let the first status line print
    Process(img);
    Cv2.ImWrite("annotated.jpg", img);
    Console.WriteLine("Saved annotated.jpg");
    return;
}

// --- live camera -----------------------------------------------------------
using var cap = new VideoCapture(camIndex);
if (!cap.IsOpened())
{
    Console.WriteLine($"Could not open camera {camIndex}. Check 'ls /dev/video*' and that the webcam is plugged in.");
    return;
}
cap.Set(VideoCaptureProperties.FrameWidth, 1280);
cap.Set(VideoCaptureProperties.FrameHeight, 720);

bool running = true;
Console.CancelKeyPress += (_, e) => { e.Cancel = true; running = false; };

using var frame = new Mat();
var lastSave = DateTime.MinValue;
Console.WriteLine("Running - press Ctrl+C to stop.");

while (running)
{
    if (!cap.Read(frame) || frame.Empty()) { Thread.Sleep(10); continue; }
    Process(frame);
    if (DateTime.UtcNow - lastSave > TimeSpan.FromSeconds(1))
    {
        Cv2.ImWrite("snapshot.jpg", frame);
        lastSave = DateTime.UtcNow;
    }
}

homography?.Dispose();
