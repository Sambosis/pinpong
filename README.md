# Pygame Video Recorder

This project now includes a reusable video recorder for Pygame: `recorder.py`.

## Features
- Simple API: `Recorder(path).capture(surface_or_none)`
- Works with `pygame.Surface` or numpy arrays
- Auto-detect display surface when `capture()` called with no args
- Sensible default FPS (uses `config.FIXED_TIMESTEP` if available, else 60)
- Auto-create output directories and timestamped filenames
- Context manager support (`with Recorder(...) as r: ...`)
- Pause/resume/toggle recording and optional frame skipping
- Optional resizing before writing (e.g., to downscale)
- Broad compatibility via `-pix_fmt yuv420p`
- Convenience context manager `record_pygame()` to auto-capture on each `pygame.display.flip()`

## Quick Start
```python
import pygame
from recorder import Recorder

pygame.init()
screen = pygame.display.set_mode((800, 600))

with Recorder("videos/demo.mp4") as rec:
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((30, 30, 30))
        # ... draw your game ...

        pygame.display.flip()
        rec.capture(screen)  # or rec.capture() to grab current display
        clock.tick(60)
```

## One-liner Auto-Capture (hooks pygame.display.flip)
```python
import pygame
from recorder import record_pygame

pygame.init()
screen = pygame.display.set_mode((800, 600))

with record_pygame("videos/run.mp4", fps=60) as rec:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # ... draw ...
        pygame.display.flip()  # auto-captured by the hook
```

## API Summary
```python
Recorder(
    path: str | None = None,   # file or directory; dir -> auto-name inside
    fps: int | None = None,    # default uses config.FIXED_TIMESTEP or 60
    *,
    codec: str = "libx264",
    quality: int = 5,          # 1=best, 10=worst (if bitrate not set)
    bitrate: str | None = None,
    macro_block_size: int = 1, # 1 allows any resolution
    ffmpeg_params: list[str] | None = None,  # defaults to ['-pix_fmt','yuv420p']
    auto_mkdir: bool = True,
    frame_skip: int = 0,       # write every (skip+1)-th frame
    start_paused: bool = False,
    resize_to: tuple[int, int] | None = None # optional (width, height)
)

rec.capture(surface_or_array=None)  # surface, ndarray, or None for display
rec.capture_frame(np.ndarray)       # same as capture(array)
rec.record_display()                # same as capture(None)
rec.pause(); rec.resume(); rec.toggle()
rec.close()

# Auto-capture hook
rec.hook_display_flip()  # context manager

# Convenience: create recorder and hook in one step
record_pygame(path=None, fps=None, **kwargs)
```

## Notes
- If `path` is a directory or `None`, files are saved to `videos/<prefix>_<timestamp>.mp4`.
- Default FPS comes from `config.FIXED_TIMESTEP` (inverse) if present, then `config.FPS`, else 60.
- Ensure `imageio` and `imageio-ffmpeg` are installed (already listed in `pyproject.toml`).
- For headless CI, you may need a dummy video driver: `os.environ['SDL_VIDEODRIVER']='dummy'` before `pygame.init()`.