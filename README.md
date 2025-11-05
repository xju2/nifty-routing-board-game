# Routing Board Game

This is a browser-based interactive implementation of the routing board puzzle.

## Screenshot

<p align="center">
  <a href="assets/gameplay.png">
    <img src="assets/gameplay.png" alt="Gameplay screenshot" width="720" />
  </a>
  <br/>
  <em>Routing board with arrows (shift‑out direction), pieces, HUD and output tile.</em>
</p>

Controls
- M: toggle Placement / Routing mode
- Left click: place/remove piece (Placement) or cycle tile direction (Routing)
- Shift + Left click: reverse-cycle tile direction (Routing)
- S: step one turn
- Space: start/stop auto-run
- R: reset board
- C: clear all pieces
- D: clear routing map
- 1..9: place that many pieces randomly
- 0: place 10 random pieces

Technical build details
```
clang --target=wasm32 -O3 -nostdlib \
  -Wl,--no-entry \
  -Wl,--export=init -Wl,--export=frame -Wl,--export=set_viewport \
  -Wl,--export=on_pointer -Wl,--export=on_key \
  -Wl,--export-memory \
  -Wl,--initial-memory=2097152 -Wl,--max-memory=16777216 \
  -Wl,--export-table \
  -Wl,--allow-undefined \
  -o main.wasm main.c
```
Open `index.html` in a local server (any static server). The canvas resizes to the window.

Examples
- macOS with MacPorts LLVM:
  - `make -C routing-board CLANG=clang-mp-19`
- Homebrew LLVM:
  - `make -C routing-board CLANG=$(brew --prefix llvm)/bin/clang`
