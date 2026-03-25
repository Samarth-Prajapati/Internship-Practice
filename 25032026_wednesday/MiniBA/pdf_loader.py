from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    """
    Load a PDF file
    Parameters
    ----------
    file_path - file path

    Returns - PDF text
    -------
    """

    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        text = ""
        for page in pages:
            text += page.page_content + "\n"

        return text

    except FileNotFoundError:
        return "PDF not found"
    except Exception as error:
        return "Error in loading PDF", error