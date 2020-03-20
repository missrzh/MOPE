import random
import tkinter as tk
from tkinter import ttk


class Main(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.init_main()

    def init_main(self):
        label_function = tk.Label(text='Y=').place(x=50, y=20)
        entry_a0 = ttk.Entry(width=2)
        entry_a0.insert(0, '0')
        entry_a0.place(x=70, y=20)

        label_function1 = tk.Label(text='+').place(x=90, y=20)
        entry_a1 = ttk.Entry(width=2)
        entry_a1.insert(0, '1')
        entry_a1.place(x=105, y=20)
        label_function1 = tk.Label(text='X1').place(x=125, y=20)

        label_function2 = tk.Label(text='+').place(x=143, y=20)
        entry_a2 = ttk.Entry(width=2)
        entry_a2.insert(0, '2')
        entry_a2.place(x=160, y=20)
        label_function3 = tk.Label(text='X2').place(x=180, y=20)

        label_function4 = tk.Label(text='+').place(x=198, y=20)
        entry_a3 = ttk.Entry(width=2)
        entry_a3.insert(0, '3')
        entry_a3.place(x=215, y=20)
        label_function5 = tk.Label(text='X3').place(x=235, y=20)

        button_generate = ttk.Button(text='Write Down and Generate Matrix', width=33)
        button_generate.place(x=50, y=47)
        button_generate.bind('<Button-1>', lambda event: self.fill_matrix(float(entry_a0.get()),float(entry_a1.get()),
                                                                          float(entry_a2.get()), float(entry_a3.get())))

        t = 0
        for i in range(8):
            label_function8 = tk.Label(text=i+1, width=3).place(x=46, y=95+t)
            t += 20
        label_function9 = tk.Label(text="x0", width=3).place(x=46, y=95+t)
        label_function9 = tk.Label(text="dx", width=3).place(x=46, y=115 + t)
        label_function9 = tk.Label(text="X1", width=3).place(x=75, y=75)
        label_function9 = tk.Label(text="X2", width=3).place(x=100, y=75)
        label_function9 = tk.Label(text="X3", width=3).place(x=125, y=75)
        label_function9 = tk.Label(text="Y", width=3).place(x=150, y=75)
        label_function9 = tk.Label(text="XH1", width=3).place(x=175, y=75)
        label_function9 = tk.Label(text="XH2", width=3).place(x=203, y=75)
        label_function9 = tk.Label(text="XH3", width=3).place(x=229, y=75)

    def fill(self):
        lis = [random.randrange(20) for k in range(8)]
        return lis

    def find_x0(self, lis):
        name = (max(lis) + min(lis)) / 2
        return name

    def find_dx(self, x0, lis):
        dx = x0 - min(lis)
        try:
            dx == max(lis) - x0
        except ValueError:
            print("Logical error")
        return dx

    def fill_XH(self, lis, x0, dx):
        XH = [(lis[k] - x0) / dx for k in range(len(lis))]
        print(XH)
        return XH

    def fill_matrix(self, a0, a1, a2, a3):
        x1 = self.fill()
        x2 = self.fill()
        x3 = self.fill()
        y = [a0 + a1 * x1[i] + a2 * x2[i] + a3 * x3[i] for i in range(8)]
        maxy = y.index(max(y))
        print(maxy)
        t = 0
        for k, i in enumerate(x1):
            if k == maxy:
                label_function8 = tk.Label(text=i + 1, width=3, background="red").place(x=75, y=95 + t)
                t += 20
            else:
                label_function8 = tk.Label(text=i + 1, width=3, background="white").place(x=75, y=95 + t)
                t += 20
        t=0
        for k,i in enumerate(x2):
            if k == maxy:
                label_function8 = tk.Label(text=i + 1, width=3, background="red").place(x=100, y=95 + t)
                t += 20
            else:
                label_function8 = tk.Label(text=i + 1, width=3, background="white").place(x=100, y=95 + t)
                t += 20
        t = 0
        for k,i in enumerate(x3):
            if k == maxy:
                label_function8 = tk.Label(text=i + 1, width=3, background="red").place(x=125, y=95 + t)
                t += 20
            else:
                label_function8 = tk.Label(text=i + 1, width=3, background="white").place(x=125, y=95 + t)
                t += 20
        label_function8 = tk.Label(text=self.find_x0(x1), width=3, background="orange").place(x=75, y=95 + t)
        label_function8 = tk.Label(text=self.find_x0(x2), width=3, background="orange").place(x=100, y=95 + t)
        label_function8 = tk.Label(text=self.find_x0(x3), width=3, background="orange").place(x=125, y=95 + t)
        t=0
        for k, i in enumerate(y):
            if k == maxy:
                label_function8 = tk.Label(text=round(i,2), width=3, background="red").place(x=150, y=95 + t)
                t += 20
            else:
                label_function8 = tk.Label(text=round(i, 2), width=3, background="grey").place(x=150, y=95 + t)
                t += 20
        x01 = self.find_x0(x1)
        x02 = self.find_x0(x2)
        x03 = self.find_x0(x3)
        ye = a0+a1*x01+a2*x02+a3*x03
        abel_function8 = tk.Label(text=ye, width=3, background="gray").place(x=150, y=95 + t)
        dx1 = self.find_dx(x01, x1)
        dx2 = self.find_dx(x02, x2)
        dx3 = self.find_dx(x03, x3)
        list_of_dx = [dx1, dx2, dx3]
        XH1 = self.fill_XH(x1, x01, dx1)
        XH2 = self.fill_XH(x2, x02, dx2)
        XH3 = self.fill_XH(x3, x03, dx3)
        list_of_xh = [XH1, XH2, XH3]
        s = 0
        for i in list_of_xh:
            k = 0
            for n in i:
                label_function8 = tk.Label(text= round(n, 2), width=3, background="yellow").place(x=177 + s, y=95 + k)
                k += 20
            s += 27
        s = 0
        for i in list_of_dx:
            label_function8 = tk.Label(text=i, width=3, background="orange").place(x=75 + s, y=115 + t)
            s += 25
        label_function8 = tk.Label(text=0, width=3, background="yellow").place(x=178, y=95 + t)
        label_function8 = tk.Label(text=0, width=3, background="yellow").place(x=205, y=95 + t)
        label_function8 = tk.Label(text=0, width=3, background="yellow").place(x=231, y=95 + t)
        ydx = a0 + a1 * dx1 + a2 * dx2 + a3 * dx3
        label_function8 = tk.Label(text=ydx, width=3, background="grey").place(x=150, y=115 + t)
        XHdx1 = self.fill_XH([dx1], x01, dx1)
        XHdx2 = self.fill_XH([dx2], x02, dx2)
        XHdx3 = self.fill_XH([dx3], x03, dx3)
        list_of_XHdx = [XHdx1, XHdx2, XHdx3]
        print(list_of_XHdx)
        s=0
        for i in list_of_XHdx:
            label_function8 = tk.Label(text=round(i[0], 2), width=3, background="yellow").place(x=178 + s, y=116 + t)
            s+=27

if __name__ == '__main__':
    root = tk.Tk()
    app = Main(root)
    app.pack()
    root.title('MOPE Lab1')
    root.geometry('300x320+100+100')
    root.resizable(False, False)
    root.mainloop()
