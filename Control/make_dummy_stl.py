# Control/make_dummy_stl.py
def make_dummy_stl(filename="Control/dummy_surface.stl"):
    with open(filename, "w") as f:
        f.write("""solid dummy
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid dummy
""")
    print(f"Created dummy STL at {filename}")

if __name__ == "__main__":
    make_dummy_stl()
