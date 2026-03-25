def prompt_loader(file_path):
    """
    Loading Prompt Markdown File
    Parameters
    ----------
    file_path - path of the file to be loaded

    Returns - data from Prompt Markdown File
    -------
    """

    try:
        with open(file_path, "r") as file:
            return file.read()

    except FileNotFoundError:
        return "File Not Found"
    except Exception as error:
        return error