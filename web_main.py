"""Start the local Astraea web interface on 127.0.0.1:8000."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("astraea.web:app", host="127.0.0.1", port=8000)
