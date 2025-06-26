import os
import numpy as np
import fitz
import matplotlib.pyplot as plt


class PDFSet:
    """
    Container for one set of pdf files (either set of full cases, or set of sheets)
    """
    def __init__(self, src: str, load_format = 'none'):
        """Finds valid PDF files from a source folder and stores list of objects and filenames

        Args:
            src (str): Source folder path
        """
        self.src = src
        self.filenames_full = os.listdir(src)

        self.filenames = []
        self.filenames_removed = []
        for fn in self.filenames_full:
            if fn.endswith(".pdf"):
                self.filenames.append(fn)
            else:
                self.filenames_removed.append(fn)
        
        self.fitz = []
        self.numpy = []
        
        if load_format == 'none':
            pass
        if load_format == 'fitz':
            for filename in self.filenames:
                path = os.path.join(src, filename)
                doc = fitz.open(path)
                self.fitz.append(doc)
            self.iter = iter(self.fitz)
        if load_format == 'numpy':
            for filename in self.filenames:
                path = os.path.join(src, filename)
                doc = fitz.open(path)
                doc_pages = []
                for page in doc:
                    pix = page.get_pixmap(dpi=150)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n) # HWC
                    doc_pages.append(img)
                self.numpy.append(doc_pages)
            self.iter = iter(self.numpy)
    
    def __len__(self):
        return len(self.filenames)
    
    def __iter__(self):
        return self.iter
    
    def get_colormaps(self, visualize = False) -> np.ndarray:
        """Find average color per page in list of case files

        Args:
            data (CaseSet): Source data containing list of list of numpy images
            visualize (bool, optional): Tool to visualize the colorma per doc and page. Defaults to False.
        """
        
        # Record average page colors (int, RGB)
        colormaps = np.ones((len(self), 200, 3), dtype=np.uint8)

        for idx, doc in enumerate(self):
            for i, img in enumerate(doc):
                avg = np.mean(img, axis=(0, 1))
                colormaps[idx, i] = avg
            
            print(f"Colormapping {idx} done")
        
        if visualize:
            plt.figure(figsize=(15, 2))
            plt.imshow(colormaps, interpolation='nearest')
            plt.title("Page average colors")
            plt.show()
        
        return colormaps
                    