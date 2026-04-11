"""Run the OpenCure web server."""

import os
import uvicorn


def main():
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "opencure.web.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
