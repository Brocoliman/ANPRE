import os
import numpy as np
import fitz
import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter
import cv2


class PDFSet:
    """
    Container for one set of pdf files (either set of full cases, or set of sheets)
    """
    
    def __init__(self, pdfs=None, imgs=None, load_format='', filepaths: list[str] = None, print_events=True):
        self.pdfs = pdfs
        self.imgs = imgs
        self.load_format = load_format
        self.filepaths = filepaths
        self.print_events = print_events
    
    @classmethod
    def reduce(cls, src, selection: list[list[int]]):
        """Takes only selection of pages from each document in PDFSet

        Args:
            src (PDFSet): source PDFSet
            selection (list[list[int]]): describes the pages to take from each document

        Returns:
            PDFSet: new PDFSet with reduced fitz documents and/or image arrays
        """

        reduced_pdfs = []
        reduced_imgs = []

        for doc_idx in range(len(src)):
            # Reduce fitz documents
            if src.load_format.find("fitz") != -1:
                full_doc = src.pdfs[doc_idx]
                new_doc = fitz.open()
                for page_num in selection[doc_idx]:
                    new_doc.insert_pdf(full_doc, from_page=page_num, to_page=page_num)
                reduced_pdfs.append(new_doc)
            
            # Reduce numpy images
            if src.load_format.find("numpy") != -1:
                full_imgs = src.imgs[doc_idx]
                new_imgs = [full_imgs[page_num] for page_num in selection[doc_idx]]
                reduced_imgs.append(new_imgs)

        return cls(
            pdfs=reduced_pdfs if src.load_format.find("fitz") != -1 else None,
            imgs=reduced_imgs if src.load_format.find("numpy") != -1 else None,
            load_format=src.load_format,
            filepaths=src.filepaths,
            print_events=src.print_events
        )


    @classmethod
    def fromFolder(cls, src: str, load_format='', print_events=True):
        """Finds valid PDF files from a source folder and stores list of objects and filepaths

        Args:
            src (str): Source folder path
            load_format (str): Whether to load PDF object data, and if as PyMuPDF object (fitz) or numpy ndarray
                - none: do not load PDF object data
                - fitz: load into fitz pdf
                - numpyN: load into array with N dpi
        
        Returns:
            PDFSet: generated from folder of PDF
        """
        
        filepaths = []
        for fn in os.listdir(src):
            if fn.endswith(".pdf"):
                filepaths.append(os.path.join(src, fn))
        
        pdfs = []
        imgs = []
        output = cls(pdfs, imgs, load_format, filepaths, print_events)
        
        if load_format.find("fitz") != -1:
            for i, filename in enumerate(output.filepaths):
                doc = fitz.open(filename)
                pdfs.append(doc)
                if output.print_events:
                    print(f"[TASK] Loading {i}: {filename} into fitz done.")
        if load_format.find("numpy") != -1:
            start = load_format.find("numpy") + 5
            end = load_format.find(' ', load_format.find("numpy") + 5)
            dpi = int(load_format[start:end])
            for i, filename in enumerate(output.filepaths):
                doc = fitz.open(filename)
                doc_pages = []
                for page in doc:
                    pix = page.get_pixmap(dpi=dpi)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )  # HWC
                    doc_pages.append(img)
                imgs.append(doc_pages)
                if print_events:
                    print(f"[TASK] Loading {i}: {filename} into numpy done.")
        
        return output
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        # To access documents
        if isinstance(idx, int):
            return self.pdfs[idx], self.imgs[idx], self.filepaths[idx]
        
        # To access pages
        if isinstance(idx, tuple):
            doc_idx, page_idx = idx
            return self.pdfs[doc_idx][page_idx], self.imgs[doc_idx][page_idx], self.filepaths[doc_idx]

    def pdf_iter(self):
        return iter(self.pdfs)
    
    def img_iter(self):
        return iter(self.imgs)

    def get_colormaps(self, visualize=False) -> np.ndarray:
        """Find average color per page in list of case files

        Args:
            data (CaseSet): Source data containing list of list of numpy images
            visualize (bool, optional): Tool to visualize the colorma per doc and page. Defaults to False.
        """

        # Record average page colors (int, RGB)
        colormaps = np.ones((len(self), 300, 3), dtype=np.uint8)

        for idx, doc in enumerate(self.img_iter()):
            for i, img in enumerate(doc):
                avg = np.mean(img, axis=(0, 1))
                colormaps[idx, i] = avg
                if i >= 300:
                    break

            print(
                f"[EVENT] Colormapping {idx} done. {'Note: report exceeded 300 pages' if len(doc)>300 else ''}"
            )

        if visualize:
            plt.figure(figsize=(15, 2))
            plt.imshow(colormaps, interpolation="nearest")
            plt.title("Page average colors")
            plt.show()

        return colormaps

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        if self.load_format.find('fitz') != -1: # from original document is preferred
            for doc_idx, (doc, filepath) in enumerate(zip(self.pdfs, self.filepaths)):
                writer = fitz.open()
                for page_num in range(len(doc)):
                    writer.insert_pdf(doc, from_page=page_num, to_page=page_num)

                if len(writer) == 0:
                    if self.print_events:
                        print(f"[SKIP] Document {doc_idx} is empty, skipped")
                    continue

                out_path = os.path.join(save_dir, f"{doc_idx}_{os.path.basename(filepath)}")
                writer.save(out_path)
                if self.print_events:
                    print(f"[SAVE] Saved fitz document {doc_idx} to {out_path}")


        elif self.load_format.find("numpy") != -1:
            for doc_idx, (pages, filepath) in enumerate(zip(self.imgs, self.filepaths)):
                if not pages:  # skip empty doc
                    if self.print_events:
                        print(f"[SKIP] Numpy document {doc_idx} empty, skipped")
                    continue

                writer = PdfWriter()

                for page_img in pages:
                    height, width = page_img.shape[:2]
                    if page_img.shape[2] == 4:
                        page_img = page_img[:, :, :3]

                    img_bytes = cv2.imencode(".jpg", page_img)[1].tobytes()

                    doc = fitz.open()
                    img_pdf = doc.new_page(width=width, height=height)
                    img_rect = fitz.Rect(0, 0, width, height)
                    img_pdf.insert_image(img_rect, stream=img_bytes)

                    pdf_bytes = doc.write()
                    reader = PdfReader()
                    reader.append_pages_from_reader(PdfReader(fitz.open("pdf", pdf_bytes)))
                    writer.add_page(reader.pages[0])

                if len(writer.pages) == 0:
                    if self.print_events:
                        print(f"[SKIP] Final output for doc {doc_idx} is empty, skipped")
                    continue

                out_path = os.path.join(save_dir, f"{doc_idx}_{os.path.basename(filepath)}")
                with open(out_path, "wb") as f:
                    writer.write(f)
                if self.print_events:
                    print(f"[SAVE] Saved numpy document {doc_idx} to {out_path}")

        else:
            raise ValueError("No data stored in this object.")