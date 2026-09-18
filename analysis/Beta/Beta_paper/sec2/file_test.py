import caesar

fname = (
    "/cosma8/data/dp376/dc-xian3/simba-eor/"
    "EoRData/FullSpectra_Fit/m25n1024/"
    "caesar_m25n1024_016_calzetti.hdf5"
)

obj = caesar.load(fname)

g = obj.galaxies[0]

print("photometry type:")
print(type(g.photometry))

print("\nphotometry:")
print(g.photometry)

print("\nphotometry keys:")
print(g.photometry.keys())

print("\nTrying beta:")
print(g.photometry["beta"])
