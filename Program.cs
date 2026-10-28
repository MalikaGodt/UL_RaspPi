using System.Buffers.Binary;
using System.Device.I2c;
using System.Numerics;
using Silk.NET.Input;
using Silk.NET.Maths;
using Silk.NET.OpenGLES;
using Silk.NET.Windowing;

// Usage:
//   dotnet run                 BNO085 at 0x4A, Game Rotation Vector (no magnetometer)
//   dotnet run -- --alt        BNO085 at 0x4B (Adafruit breakout with the address jumper set)
//   dotnet run -- --rv         Rotation Vector (uses magnetometer -> absolute heading)
//   dotnet run -- --sim        no sensor, fake motion (test the 3D window first)

bool sim = args.Contains("--sim");
int addr = args.Contains("--alt") ? 0x4B : 0x4A;
byte report = args.Contains("--rv") ? Bno085.RotationVector : Bno085.GameRotationVector;

Quaternion latest = Quaternion.Identity;
var gate = new object();
var cts = new CancellationTokenSource();

Bno085? imu = null;
if (!sim)
{
    imu = new Bno085(busId: 1, address: addr);
    imu.Begin(report, intervalMs: 10);   // 100 Hz
}

var reader = Task.Run(() =>
{
    var t0 = DateTime.UtcNow;
    while (!cts.IsCancellationRequested)
    {
        Quaternion q;
        if (sim)
        {
            double t = (DateTime.UtcNow - t0).TotalSeconds;
            q = Quaternion.CreateFromYawPitchRoll((float)(t * 0.8), (float)(Math.Sin(t) * 0.5), (float)(Math.Sin(t * 0.7) * 0.7));
            Thread.Sleep(10);
        }
        else if (!imu!.TryReadOrientation(out q))
        {
            Thread.Sleep(1);
            continue;
        }
        lock (gate) latest = q;
    }
});

new Viewer(() => { lock (gate) { return latest; } }).Run();

cts.Cancel();
reader.Wait();
imu?.Dispose();

// ---------------------------------------------------------------------------
// Minimal BNO085 driver (SHTP over I2C) - only what is needed for orientation
// ---------------------------------------------------------------------------
sealed class Bno085 : IDisposable
{
    public const byte GameRotationVector = 0x08;
    public const byte RotationVector = 0x05;

    const byte ChExecutable = 1, ChControl = 2, ChSensorReports = 3;

    readonly I2cDevice _i2c;
    readonly byte[] _seq = new byte[8];
    readonly byte[] _buf = new byte[1024];

    public Bno085(int busId = 1, int address = 0x4A)
        => _i2c = I2cDevice.Create(new I2cConnectionSettings(busId, address));

    public void Begin(byte reportId, int intervalMs)
    {
        Send(ChExecutable, new byte[] { 1 });   // soft reset
        Thread.Sleep(300);
        Drain();                                // throw away advertisement / reset-complete packets

        // SH-2 "Set Feature Command" (17 bytes)
        var p = new byte[17];
        p[0] = 0xFD;
        p[1] = reportId;
        BinaryPrimitives.WriteUInt32LittleEndian(p.AsSpan(5), (uint)(intervalMs * 1000));
        Send(ChControl, p);
    }

    public bool TryReadOrientation(out Quaternion q)
    {
        q = default;
        if (!ReadPacket(out int len, out byte ch) || ch != ChSensorReports) return false;

        bool got = false;
        int o = 4;                              // skip SHTP header
        while (o < len)
        {
            switch (_buf[o])
            {
                case 0xFB: case 0xFA: o += 5; break;                       // timestamp reports
                case GameRotationVector: q = Parse(o + 4); got = true; o += 12; break;
                case RotationVector:     q = Parse(o + 4); got = true; o += 14; break;
                default: return got;
            }
        }
        return got;
    }

    Quaternion Parse(int o)
    {
        const float s = 1f / 16384f;            // Q14 fixed point
        float i = BinaryPrimitives.ReadInt16LittleEndian(_buf.AsSpan(o)) * s;
        float j = BinaryPrimitives.ReadInt16LittleEndian(_buf.AsSpan(o + 2)) * s;
        float k = BinaryPrimitives.ReadInt16LittleEndian(_buf.AsSpan(o + 4)) * s;
        float r = BinaryPrimitives.ReadInt16LittleEndian(_buf.AsSpan(o + 6)) * s;
        return Quaternion.Normalize(new Quaternion(i, j, k, r));
    }

    bool ReadPacket(out int len, out byte channel)
    {
        len = 0; channel = 0;
        Span<byte> hdr = stackalloc byte[4];
        try
        {
            _i2c.Read(hdr);
            int total = (hdr[0] | (hdr[1] << 8)) & 0x7FFF;
            if (total <= 4 || total > 2048) return false;   // 0 = nothing waiting
            total = Math.Min(total, _buf.Length);
            _i2c.Read(_buf.AsSpan(0, total));               // sensor re-sends the whole packet
            channel = _buf[2];
            len = total;
            return true;
        }
        catch (IOException) { return false; }
    }

    void Drain()
    {
        var end = DateTime.UtcNow.AddMilliseconds(600);
        while (DateTime.UtcNow < end)
            if (!ReadPacket(out _, out _)) Thread.Sleep(20);
    }

    void Send(byte channel, byte[] payload)
    {
        int total = payload.Length + 4;
        var pkt = new byte[total];
        pkt[0] = (byte)(total & 0xFF);
        pkt[1] = (byte)(total >> 8);
        pkt[2] = channel;
        pkt[3] = _seq[channel]++;
        payload.CopyTo(pkt, 4);
        _i2c.Write(pkt);
    }

    public void Dispose() => _i2c.Dispose();
}

// ---------------------------------------------------------------------------
// 3D viewer: a board that rotates with the sensor + body axes + fixed grid
// ---------------------------------------------------------------------------
unsafe sealed class Viewer
{
    const string Vert = @"
attribute vec3 aPos; attribute vec3 aCol; uniform mat4 uMvp; varying vec3 vCol;
void main() { vCol = aCol; gl_Position = uMvp * vec4(aPos, 1.0); }";

    const string Frag = @"
precision mediump float; varying vec3 vCol;
void main() { gl_FragColor = vec4(vCol, 1.0); }";

    readonly Func<Quaternion> _getQ;
    IWindow _win = null!;
    GL _gl = null!;
    uint _prog, _vbo;
    int _uMvp;
    int _boxVerts, _axisVerts, _gridVerts;
    double _acc;
    Quaternion _mount = Quaternion.Identity;   // how the sensor is mounted relative to the drawn board
    Quaternion _zero = Quaternion.Identity;    // "zero" pose set with Space
    bool _invert;                              // flip rotation direction

    public Viewer(Func<Quaternion> getQ) => _getQ = getQ;

    public void Run()
    {
        var opts = WindowOptions.Default;
        opts.Size = new Vector2D<int>(900, 700);
        opts.Title = "BNO085 3D";
        opts.VSync = true;
        opts.API = new GraphicsAPI(ContextAPI.OpenGLES, ContextProfile.Core, ContextFlags.Default, new APIVersion(2, 0));

        _win = Window.Create(opts);
        _win.Load += OnLoad;
        _win.Render += OnRender;
        _win.Run();
    }

    void OnLoad()
    {
        _gl = GL.GetApi(_win);
        _gl.Enable(EnableCap.DepthTest);
        _gl.ClearColor(0.08f, 0.09f, 0.11f, 1f);

        var input = _win.CreateInput();
        foreach (var kb in input.Keyboards) kb.KeyDown += OnKeyDown;

        uint vs = Compile(ShaderType.VertexShader, Vert);
        uint fs = Compile(ShaderType.FragmentShader, Frag);
        _prog = _gl.CreateProgram();
        _gl.AttachShader(_prog, vs);
        _gl.AttachShader(_prog, fs);
        _gl.BindAttribLocation(_prog, 0, "aPos");
        _gl.BindAttribLocation(_prog, 1, "aCol");
        _gl.LinkProgram(_prog);
        _gl.GetProgram(_prog, ProgramPropertyARB.LinkStatus, out int ok);
        if (ok == 0) throw new Exception(_gl.GetProgramInfoLog(_prog));
        _uMvp = _gl.GetUniformLocation(_prog, "uMvp");

        float[] data = BuildGeometry().ToArray();
        _vbo = _gl.GenBuffer();
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);
        _gl.BufferData<float>(BufferTargetARB.ArrayBuffer, (ReadOnlySpan<float>)data, BufferUsageARB.StaticDraw);
    }

    void OnRender(double dt)
    {
        Quaternion q = Orient(_getQ());
        var size = _win.FramebufferSize;
        _gl.Viewport(0, 0, (uint)size.X, (uint)size.Y);
        _gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

        // Z-up world, camera looking at the origin
        var proj = Perspective(MathF.PI / 4f, size.X / (float)Math.Max(1, size.Y), 0.1f, 50f);
        var view = Matrix4x4.CreateLookAt(new Vector3(1.5f, -6f, 2.5f), Vector3.Zero, Vector3.UnitZ);
        var model = Matrix4x4.CreateFromQuaternion(q);

        _gl.UseProgram(_prog);
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, _vbo);
        _gl.EnableVertexAttribArray(0);
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 24, (void*)0);
        _gl.EnableVertexAttribArray(1);
        _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 24, (void*)12);

        // fixed grid
        SetMvp(view * proj);
        _gl.DrawArrays(PrimitiveType.Lines, _boxVerts + _axisVerts, (uint)_gridVerts);

        // rotating board + its own XYZ axes (X red, Y green, Z blue)
        SetMvp(model * view * proj);
        _gl.DrawArrays(PrimitiveType.Triangles, 0, (uint)_boxVerts);
        _gl.DrawArrays(PrimitiveType.Lines, _boxVerts, (uint)_axisVerts);

        _acc += dt;
        if (_acc > 0.2)
        {
            _acc = 0;
            var (roll, pitch, yaw) = ToEulerDeg(q);
             _win.Title = $"X red {roll:+0;-0;0}°   |   Y green {pitch:+0;-0;0}°   |   Z blue {yaw:+0;-0;0}°";
        }
    }

    // Keys: X/Y/Z turn the board 90 deg about its own axis (fixes a different sensor mounting),
    // C inverts the rotation direction, Space zeroes the current pose, R resets all of it.
    void OnKeyDown(IKeyboard kb, Key key, int scancode)
    {
        switch (key)
        {
            case Key.X: TurnMount(Vector3.UnitX); break;
            case Key.Y: TurnMount(Vector3.UnitY); break;
            case Key.Z: TurnMount(Vector3.UnitZ); break;
            case Key.C: _invert = !_invert; break;
            case Key.Space: _zero = Quaternion.Conjugate(Body(_getQ())); break;
            case Key.R: _mount = Quaternion.Identity; _zero = Quaternion.Identity; _invert = false; break;
        }
    }

    void TurnMount(Vector3 axis)
        => _mount = Quaternion.Normalize(Quaternion.Concatenate(Quaternion.CreateFromAxisAngle(axis, MathF.PI / 2f), _mount));

    Quaternion Body(Quaternion raw) => Quaternion.Concatenate(_mount, _invert ? Quaternion.Conjugate(raw) : raw);

    Quaternion Orient(Quaternion raw) => Quaternion.Concatenate(Body(raw), _zero);

    void SetMvp(Matrix4x4 m) => _gl.UniformMatrix4(_uMvp, 1, false, (float*)&m);

    uint Compile(ShaderType type, string src)
    {
        uint s = _gl.CreateShader(type);
        _gl.ShaderSource(s, src);
        _gl.CompileShader(s);
        _gl.GetShader(s, ShaderParameterName.CompileStatus, out int ok);
        if (ok == 0) throw new Exception(_gl.GetShaderInfoLog(s));
        return s;
    }

    // OpenGL-style perspective, laid out for System.Numerics' row-vector convention
    static Matrix4x4 Perspective(float fov, float aspect, float near, float far)
    {
        float f = 1f / MathF.Tan(fov / 2f);
        return new Matrix4x4(
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), -1,
            0, 0, 2 * far * near / (near - far), 0);
    }

    
    static (float roll, float pitch, float yaw) ToEulerDeg(Quaternion q)
    {
        const float d = 180f / MathF.PI;
        float roll = MathF.Atan2(2 * (q.W * q.X + q.Y * q.Z), 1 - 2 * (q.X * q.X + q.Y * q.Y));
        float pitch = MathF.Asin(Math.Clamp(2 * (q.W * q.Y - q.Z * q.X), -1f, 1f));
        float yaw = MathF.Atan2(2 * (q.W * q.Z + q.X * q.Y), 1 - 2 * (q.Y * q.Y + q.Z * q.Z));
        return (roll * d, pitch * d, yaw * d);
    }

    List<float> BuildGeometry()
    {
        var v = new List<float>();

        // --- board (flat box) ---
        const float hx = 1.0f, hy = 0.6f, hz = 0.12f;
        Vector3 C(float x, float y, float z) => new(x * hx, y * hy, z * hz);
        void Quad(Vector3 a, Vector3 b, Vector3 c, Vector3 d, Vector3 col)
        {
            foreach (var p in new[] { a, b, c, a, c, d })
                v.AddRange(new[] { p.X, p.Y, p.Z, col.X, col.Y, col.Z });
        }
        Quad(C(-1, -1, 1), C(1, -1, 1), C(1, 1, 1), C(-1, 1, 1), new(0.25f, 0.75f, 0.35f));      // +Z top
        Quad(C(-1, -1, -1), C(-1, 1, -1), C(1, 1, -1), C(1, -1, -1), new(0.10f, 0.35f, 0.15f));  // -Z
        Quad(C(1, -1, -1), C(1, 1, -1), C(1, 1, 1), C(1, -1, 1), new(0.85f, 0.35f, 0.35f));      // +X
        Quad(C(-1, -1, -1), C(-1, -1, 1), C(-1, 1, 1), C(-1, 1, -1), new(0.45f, 0.15f, 0.15f));  // -X
        Quad(C(-1, 1, -1), C(-1, 1, 1), C(1, 1, 1), C(1, 1, -1), new(0.35f, 0.45f, 0.90f));      // +Y (front)
        Quad(C(-1, -1, -1), C(1, -1, -1), C(1, -1, 1), C(-1, -1, 1), new(0.15f, 0.20f, 0.50f));  // -Y
        _boxVerts = 36;

        // --- body axes ---
        void Line(Vector3 a, Vector3 b, Vector3 col)
        {
            v.AddRange(new[] { a.X, a.Y, a.Z, col.X, col.Y, col.Z });
            v.AddRange(new[] { b.X, b.Y, b.Z, col.X, col.Y, col.Z });
        }
        Line(Vector3.Zero, new(1.8f, 0, 0), new(1f, 0.15f, 0.15f));
        Line(Vector3.Zero, new(0, 1.8f, 0), new(0.2f, 1f, 0.2f));
        Line(Vector3.Zero, new(0, 0, 1.8f), new(0.3f, 0.5f, 1f));
        _axisVerts = 6;

        // --- fixed floor grid ---
        for (int i = -4; i <= 4; i++)
        {
            float t = i * 0.5f;
            Line(new(t, -2, -1.4f), new(t, 2, -1.4f), new(0.28f, 0.30f, 0.34f));
            Line(new(-2, t, -1.4f), new(2, t, -1.4f), new(0.28f, 0.30f, 0.34f));
        }
        _gridVerts = 9 * 2 * 2;

        return v;
    }
}
