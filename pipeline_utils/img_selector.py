    
from tkinter import filedialog
from tkinter import *

class img_selector:
    """
    A class for selecting image files through either a provided path or a file dialog.
    This class provides functionality to handle image selection, particularly focusing
    on WEBP file formats. It can work with predefined paths or open a file dialog
    for interactive selection.
    Attributes:
        path (str, optional): The file path to an image. Defaults to None.
    Example:
        >>> selector = img_selector()\n
        >>> image_path = selector.select_image()
        ### Or with a predefined path
        >>> selector = img_selector("/path/to/image.webp")\n
        >>> image_path = selector.select_image()
    """
    
    def __init__(self, path=None):
        self.path = path

    def select_image(self):
            """
            Select an image either from the predefined path or through a file dialog.
            If a path was provided during initialization, returns that path.
            Otherwise, opens a file dialog for the user to select a WEBP image file.
            Returns:
                str: The path to the selected image file.
            Note:
                When using the file dialog:
                - Only WEBP files can be selected
                - The dialog window will appear on top of other windows
                - The tkinter root window is automatically cleaned up after selection
            """
            if self.path is not None:
                print(f"Selected image path: {self.path}")
                return self.path
    
            # Create and hide the main tkinter window
            root = Tk()
            root.withdraw()

            # Make the window appear on top of other windows
            root.attributes('-topmost', True)

            # Open file dialog for image selection
            image_path = filedialog.askopenfilename(
                title='Select an image file',
                filetypes=[
                    ('WEBP files', '*.webp')
                ],
                parent=root  # Set parent window to ensure proper stacking
            )

            # Clean up the root window
            root.destroy()

            # Print selected path (optional)
            print(f"Selected image path: {image_path}")
            return image_path