# Rectangle and cuboid checker

This program processes a text file containing either 4 2D coordinates or 5 3D coordinates. The coordinates should be comma-separated for axes and newline-separated for points.

## Requirements
- Python 3.11 or newer
- No external libraries are used

## Usage

### Running the Program
To run the program, execute the following command:

```
python rect_check.py [file_path]
```

If no file path is provided, the program will use `coordinates.txt` as the default file.

### Input File Format
- For 4 2D coordinates: Provide 4 lines with 2 comma-separated values for each coordinate.
- For 5 3D coordinates: Provide 5 lines with 3 comma-separated values for each coordinate.

### Output
- For 4 2D coordinates: Checks if the first three coordinates can form a rectangle and whether the fourth coordinate lies inside this rectangle.
- For 5 3D coordinates: Verifies if the first three coordinates form a rectangle, if the fourth can complete a rectangular cuboid with this rectangle, and if the fifth point is within the cuboid.

## Example
Example text file content for 4 2D coordinates:
```
-5, 1
3, 5
5, 1
3, 4
```

Example text file content for 5 3D coordinates:
```
-5, 1, -1
3, 5, -1
5, 1, -1
-3, -3, 2
0, 0, 0
```
