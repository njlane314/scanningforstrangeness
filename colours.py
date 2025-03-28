import ROOT

# Create a canvas and a multi-graph
canvas = ROOT.TCanvas("c", "Canvas", 800, 600)
mg = ROOT.TMultiGraph()
N = 10
colors = []

# Add graphs to the multi-graph
for i in range(N):
    gr = ROOT.TGraph()
    gr.SetPoint(0, 0, 0)  # Dummy points
    gr.SetPoint(1, 1, 1)
    mg.Add(gr)

# Draw the multi-graph to assign colors
mg.Draw("APL")  # A = axis, P = points, L = lines
canvas.Update()

# Extract the colors
for gr in mg.GetListOfGraphs():
    colors.append(gr.GetLineColor())

print("Natural ROOT colors (indices):", colors)