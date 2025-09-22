This is a simple Tkinter + OpenCV image tool with two modes: Color Vision and Highlight.
In Color Vision mode, you pick a color-vision type (e.g., deuteranopia/protanopia/tritanopia/achromatopsia) and either simulate how it looks or assist (daltonize) to improve color separation, with severity/strength controls.
In Highlight mode, you isolate colors using HSV sliders and a click-to-pick sampler, with options to keep only the selection, dim the background, or overlay a tint.
The viewer is resizable and shows a live preview while keeping the original image resolution.
Nothing is saved unless you press S (Save As), and saving performs a pixels→binary→pixels round-trip before writing a new file—your original is never overwritten.
