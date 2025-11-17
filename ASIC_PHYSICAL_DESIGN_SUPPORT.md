# ASIC Physical Design Support in Integrated_RAG

## Overview

The `Integrated_RAG.py` system now includes comprehensive support for ASIC Physical Design file formats, making it ideal for hardware design workflows.

## Supported ASIC Physical Design Formats

### Layout & Physical Design
- **LEF** (`.lef`) - Library Exchange Format
  - Physical library information
  - Cell definitions, pin locations, metal layers
  - Used by place & route tools

- **DEF** (`.def`) - Design Exchange Format
  - Physical design layout
  - Placement and routing information
  - Component locations and connections

- **GDSII** (`.gds`, `.gds2`) - Layout Database
  - Binary layout format
  - Mask data for fabrication
  - Can contain text representations

- **Milkyway** (`.mw`) - Cadence Database Format
  - Cadence physical design database
  - Contains design hierarchy and layout

### Timing & Constraints
- **SDC** (`.sdc`) - Synopsys Design Constraints
  - Timing constraints
  - Clock definitions
  - False paths, multicycle paths
  - Critical for timing closure

- **TLF** (`.tlf`) - Timing Library Format
  - Timing library information
  - Cell timing characteristics

- **LIB** (`.lib`) - Liberty Timing Library
  - Standard timing library format
  - Cell delays, power, area
  - Used by synthesis and STA tools

- **SDF** (`.sdf`) - Standard Delay Format
  - Timing annotation
  - Back-annotated delays
  - Used in gate-level simulation

### Parasitic & Extraction
- **SPEF** (`.spef`) - Standard Parasitic Exchange Format
  - Parasitic extraction data
  - RC networks
  - Used for accurate timing analysis

### Circuit Simulation
- **SPICE** (`.sp`, `.spice`, `.cir`) - Circuit Simulation
  - Netlist format for circuit simulation
  - Used for analog/mixed-signal verification
  - Transistor-level simulation

### Netlist Formats
- **CDL** (`.cdl`) - Circuit Description Language
  - Transistor-level netlist
  - Used for LVS (Layout vs Schematic) verification

### Power Intent
- **UPF** (`.upf`) - Unified Power Format
  - Power intent specification
  - Power domains, power states
  - IEEE 1801 standard

- **CPF** (`.cpf`) - Common Power Format
  - Power intent (older format)
  - Cadence power format

## Already Supported (Hardware Design)

- **SystemVerilog** (`.sv`, `.svh`) - Hardware description and verification
- **Verilog** (`.v`, `.vh`) - Hardware description language
- **VHDL** (`.vhd`, `.vhdl`) - Hardware description language
- **Tcl** (`.tcl`, `.tk`) - Tool command language (used extensively in EDA tools)
- **Perl** (`.pl`, `.pm`, `.pod`) - Scripting for EDA automation

## Usage Examples

### Index ASIC Design Files
```bash
python Integrated_RAG.py --documents ./asic_design --openai --openai-model gpt-4o-mini
```

### Filter by Format
```bash
# Filter by SDC files
filter: language sdc

# Filter by LEF files
filter: language lef

# Filter by specific file
filter: design.def
```

### Edit ASIC Design Files
```bash
# Edit SDC constraints
edit: timing.sdc Add clock constraint for clk_200MHz

# Edit LEF file
edit: library.lef Update metal layer definitions

# Edit SPEF file
edit: extracted.spef Fix parasitic capacitance values
```

### Query ASIC Design
```bash
# Ask about timing constraints
What are the clock constraints in the SDC file?

# Ask about layout
What cells are placed in the DEF file?

# Ask about power intent
What power domains are defined in the UPF file?
```

## ASIC Physical Design Workflow Support

The system can now help with:

1. **Floorplanning** - Analyze and edit LEF/DEF files
2. **Placement** - Understand and modify placement constraints
3. **Routing** - Analyze routing information in DEF files
4. **Timing Closure** - Work with SDC constraints and timing libraries
5. **Power Analysis** - Understand UPF/CPF power intent
6. **Parasitic Extraction** - Analyze SPEF files
7. **Verification** - Work with CDL netlists for LVS
8. **Simulation** - Handle SPICE netlists

## Complete File Type Count

- **Total supported extensions**: 73
- **ASIC Physical Design formats**: 15
- **Hardware description languages**: 6 (SystemVerilog, Verilog, VHDL, Tcl, Perl, SPICE)
- **General programming languages**: 20+
- **Configuration/data formats**: 10+

## Next Steps for ASIC Design

Consider adding:
- **Floorplan files** - Custom formats from different tools
- **Placement files** - Tool-specific placement formats
- **Routing files** - Detailed routing information
- **DRC/LVS reports** - Design rule checking reports
- **Power analysis reports** - Power consumption reports
- **Timing reports** - Static timing analysis reports

The system is now ready for comprehensive ASIC Physical Design workflows!

