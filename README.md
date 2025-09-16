# CPAC-Experimental-Logger
An app with a focus making logging experimental data easy and inituative

<img width="800" alt="Table" src="https://github.com/user-attachments/assets/ab595d29-3c7e-4617-8412-d0d6465a7dc6" />

<img width="800" height="700" alt="graph" src="https://github.com/user-attachments/assets/08aa2266-eee4-4da8-9c38-e94a5f910217" />

## How it Works
Using the settings on the left you can specify the name, units, number of repeats, and the uncertainty of both the independent and dependent variable. If you need to raise all the measurements to a power you can also specify the exponent too. In graph options you need to write in a formula for the gradient and the variable you want to solve for. For example, for Hooks Law: $F=ke$ the gradient is equal to $k$ so you should write $k$ in the **Gradient Formula** section and in the **Solve For** section. If the formula is more complex like where you investigate the frequency of a sting $f$ against its tension $T$: 

$$f^2=\frac{1}{4L^2\mu}{T}$$

Then you can write in $1/(4L^2\mu)$ in the **Gradient Formula** section and choose $\mu$ or $L$ in **Solve For** section.

Once you've finished all your settings you can enter all your values into the table. The software will take care of unit conversions and prompt you for any constant values in your formula that need to be specified. Once you press generate graph, another window will open where a line of best fit is plotted along with error bars and a max and min gradient in dotted lines. Addtionally, the final value you specified will be calculated with uncertainies given.
