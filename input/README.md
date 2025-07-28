# Input Directory

Place your PDF documents in this directory for processing.

## Usage

1. Copy your PDF files to this directory
2. Run the main system with appropriate persona and job descriptions
3. Results will be saved to the output directory

## Example

```bash
# Copy PDFs to input directory
cp document1.pdf document2.pdf ./input/

# Run analysis
python main.py --persona "HR professional" --job "Create and manage fillable forms"
```

## Supported Files

- PDF documents (any size, any complexity)
- 3-10 documents recommended for optimal processing
- Complex layouts, tables, and figures are supported

## Directory Structure

```
input/
├── document1.pdf
├── document2.pdf
├── document3.pdf
└── README.md (this file)
```
