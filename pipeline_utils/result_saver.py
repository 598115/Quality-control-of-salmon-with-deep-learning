
import os
import pandas as pd
import datetime

class ResultSaver:
    """A class for saving segmentation results.
    This class handles saving processed segmentation results to CSV files and creating standardized
    result dictionaries for image analysis data.
    Attributes:
        save_path (str): The directory path where results will be saved (defaults to current working directory)
        csv_filename (str): Name of the CSV file where results will be stored (defaults to 'results.csv')
        full_csv_path (str): Complete path to the CSV file
        image_path (str): Path to the image being processed
    Example:
        >>> saver = ResultSaver('path/to/image.jpg')
        >>> data = {'melanin': 0.5, 'melanin_location': 'center', 
        ...         'blood': 0.3, 'blood_location': 'edge'}
        >>> result_dict = saver.createResultDict(True, data)
        >>> saver.saveToCSV(result_dict)
    """

    def __init__(self, image_path):
        self.save_path = os.getcwd()
        self.csv_filename = "results.csv"
        self.full_csv_path = os.path.join(self.save_path, self.csv_filename)
        self.image_path = image_path

    def saveToCSV(self, data: dict):
        """
        Saves the provided data dictionary to a CSV file called results.csv.\n
        The CSV file is created in the specified self.save_path directory.\n
        The dict can be generated from the createResultDict function.

        Args:
            - data (dict): A dictionary containing the results of the segmentation process.
        """
        # Ensure the directory exists (though os.getcwd() usually exists)
        # This is more critical if save_path was a different directory
        os.makedirs(self.save_path, exist_ok=True)

        # Create a DataFrame from the dictionary
        df = pd.DataFrame([data])

        # Check if the file exists to decide whether to write the header
        file_exists = os.path.exists(self.full_csv_path)

        # Append the DataFrame to the CSV
        # mode='a' is for append
        # header=False prevents writing the header on subsequent appends
        # index=False prevents writing the DataFrame index as a column
        df.to_csv(self.full_csv_path, mode='a', header=not file_exists, index=False)

    def createResultDict(self, img_approved: bool, data: dict):
        """
        Creates a standardized result dictionary from the results of the segmentation mode. 
        The results can be gained from the pipeline_prediction function of the ResultUtil class. \n
        To get a resultdict of a denied image, set img_approved to False and the data dictionary to None.

        The dictionary contains the following keys:
            - image_approved: Boolean indicating if the image was approved
            - date: Date of the result in YYYY-MM-DD format
            - img: Name of the image file (without path)
            - melanin: Melanin value (if approved, else None)
            - melanin_location: Location of melanin (if approved, else None)
            - blood: Blood value (if approved, else None)
            - blood_location: Location of blood (if approved, else None)

        Args:
            img_approved (bool): Indicates if the image was approved or not.
            data (dict): Dictionary containing segmentation results. Should contain keys:
                - melanin
                - melanin_location
                - blood
                - blood_location
            these can be generated from the pipeline_prediction function of the ResultUtil class.
        returns:
            dict: A dictionary containing the results of the segmentation process.
        """

        result_dict = {
            "image_approved": img_approved,
            "date":  datetime.datetime.now().strftime("%Y-%m-%d"),
            "img": os.path.basename(self.image_path),
            "melanin": data["melanin"] if img_approved else None,
            "melanin_location": data["melanin_location"] if img_approved else None,
            "blood": data["blood"] if img_approved else None,
            "blood_location": data["blood_location"] if img_approved else None
        }
        if img_approved:
            print("Image segmentation results saved to results.csv")
        else:
            print("Rejected image logged to results.csv")
        return result_dict