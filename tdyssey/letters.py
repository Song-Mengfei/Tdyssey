"""Versioned, edge-connected block capitals; no font downloads or raster assets."""

from .geometry import Rectangle, Shape

# Five columns by seven rows. Staircase diagonals share edges rather than
# fragile corner-only joins. T/L retain their separately specified exact shapes.
PATTERNS = {
    "A": ("01110", "11011", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10011", "10011", "11110", "10011", "10011", "11110"),
    "C": ("01111", "11000", "10000", "10000", "10000", "11000", "01111"),
    "D": ("11110", "10011", "10001", "10001", "10001", "10011", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "11000", "10000", "10111", "10001", "11011", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00001", "00001", "00001", "10001", "11011", "01110"),
    "K": ("10011", "10110", "11100", "11000", "11100", "10110", "10011"),
    "M": ("11011", "11111", "10101", "10001", "10001", "10001", "10001"),
    "N": ("11001", "11101", "10101", "10111", "10011", "10001", "10001"),
    "O": ("01110", "11011", "10001", "10001", "10001", "11011", "01110"),
    "P": ("11110", "10011", "10011", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "11011", "10001", "10001", "10101", "11111", "01111"),
    "R": ("11110", "10011", "10011", "11110", "11100", "10110", "10011"),
    "S": ("01111", "11000", "11000", "01110", "00011", "00011", "11110"),
    "U": ("10001", "10001", "10001", "10001", "10001", "11011", "01110"),
    "V": ("10001", "10001", "10001", "11011", "01010", "01110", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11111", "11011"),
    "X": ("11011", "01110", "00100", "00100", "00100", "01110", "11011"),
    "Y": ("11011", "01110", "00100", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00011", "00110", "00100", "01100", "11000", "11111"),
}
ALPHABET = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ALL_SHAPES = (*ALPHABET, "T_narrow")


def block_letter(name):
    """120 high, 100 wide, 20 x 120/7 cells, mass-centered coordinates."""
    rows = PATTERNS[name]
    rectangles = []
    for y, row in enumerate(rows):
        x = 0
        while x < 5:
            if row[x] == "0":
                x += 1
                continue
            start = x
            while x < 5 and row[x] == "1":
                x += 1
            rectangles.append(Rectangle(start * 20, y * 120 / 7, x * 20, (y + 1) * 120 / 7))
    area = sum(r.area for r in rectangles)
    cx, cy = (sum(r.area * r.center[i] for r in rectangles) / area for i in (0, 1))
    centered = tuple(Rectangle(r.x0-cx, r.y0-cy, r.x1-cx, r.y1-cy) for r in rectangles)
    symmetry = 2 if rows == tuple(row[::-1] for row in rows[::-1]) else 1
    return Shape(name, centered, symmetry, "official_block_capital_v1_area_centroid")
