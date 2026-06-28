print("[startup] Importing dashboard application…", flush=True)
import os

from castle_dashboard.app import create_app
from castle_dashboard.services.startup_manager import startup_manager

print("[startup] Building Dash layout and callbacks…", flush=True)
app = create_app()
server = app.server
print("[startup] Web application created.", flush=True)

if __name__ == "__main__":
    host = os.getenv("DASH_HOST", "127.0.0.1")
    port = int(os.getenv("DASH_PORT", "13209"))
    print(f"[startup] Starting web server on http://{host}:{port}", flush=True)
    startup_manager.start()
    app.run(debug=False, host=host, port=port)
