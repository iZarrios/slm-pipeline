import logging
import pdfplumber

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_pdf_with_layout(path: str) -> str:
    """
    Extracts text from a PDF while preserving layout using pdfplumber's built-in feature.

    Args:
        path (str): The file path to the PDF document.

    Returns:
        str: The extracted text, with layout awareness.
    """
    # NOTE: we can make this better by using neural models (e.g. LayoutParser, chunkr)
    all_text = []
    logging.info(f"Starting PDF extraction for: {path}")

    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                # The 'layout=True' argument tells pdfplumber to use a layout-aware
                # algorithm to determine the order of the text, rather than the
                # raw, unsorted order from the PDF file.
                text = page.extract_text(layout=True)
                if text:
                    all_text.append(text)
        logging.info(f"Successfully extracted {len(all_text)} pages from: {path}")

    except Exception as e:
        logging.error(f"Error processing PDF at {path}: {e}")
        return f"Error processing PDF: {e}"

    return "\n".join(all_text)
