# Draw.io Diagrams

This folder contains architecture diagrams for the Quant Analytics application.

## Files

| File | Description |
|------|-------------|
| `system_architecture.drawio` | High-level system architecture showing all components |
| `data_flow.drawio` | Data flow from WebSocket to UI with timing and formulas |

## How to View/Edit

1. **Online**: Go to [draw.io](https://app.diagrams.net/) and open the `.drawio` file
2. **VS Code**: Install the "Draw.io Integration" extension
3. **Desktop**: Download [draw.io Desktop](https://github.com/jgraph/drawio-desktop/releases)

## Exporting

To export as PNG/SVG:
1. Open in draw.io
2. File → Export as → PNG/SVG
3. Save to `drawio/exports/` folder

## Diagram Contents

### System Architecture
- Binance WebSocket connection
- Backend components (ingest, buffer, analytics, storage)
- Frontend components (Streamlit, session state)
- Data flow arrows with color coding

### Data Flow
- 5-step pipeline visualization
- Timing constraints for each stage
- Analytics formulas reference
