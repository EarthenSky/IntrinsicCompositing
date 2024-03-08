import os
from platform import uname
from tkinter import Tk, ttk

root = Tk()
frm = ttk.Frame(root, padding=10)
frm.grid()

ttk.Label(frm, text=f"This window is running from {uname().system}").grid(column=0, row=0)
ttk.Button(frm, text="Quit", command=root.destroy).grid(column=0, row=1)
root.mainloop()