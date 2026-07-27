Here's the full command list. Given the four algorithm definitions and the two datasets, this covers running everything and then producing the plots/website.

1. Run the benchmarks

# SQ variant — dimension-agnostic, works on both datasets (you've done these)
python run.py --local --algorithm arangodb-ivf-hnsw-sq --dataset glove-100-angular DONE
python run.py --local --algorithm arangodb-ivf-hnsw-sq --dataset sift-128-euclidean DONE

# Plain IVF-flat baseline — also works on both
python run.py --local --algorithm arangodb --dataset glove-100-angular DONE
python run.py --local --algorithm arangodb --dataset sift-128-euclidean DONE

# PQ variants — dataset-specific (PQ<m> must divide the dimension)
python run.py --local --algorithm arangodb-ivf-hnsw-pq-glove --dataset glove-100-angular
python run.py --local --algorithm arangodb-ivf-hnsw-pq-sift  --dataset sift-128-euclidean

2. Plot the results

python plot.py --dataset glove-100-angular --x-scale logit --y-scale log
python plot.py --dataset sift-128-euclidean --x-scale logit --y-scale log

3. Build the website

python create_website.py --plottype recall/time --latex --scatter --outputdir website/

A couple of notes:
- If you only care about the SQ variant you already ran, you can skip the arangodb, arangodb-ivf-hnsw-pq-glove, and arangodb-ivf-hnsw-pq-sift lines — but including all four gives you comparison curves in the plots.
- The PQ algorithm is intentionally split into -glove and -sift because PQ<m> requires m to divide the vector dimension (100 vs 128), so don't cross them.

Want me to save this workflow (and the ArangoDB integration context) to my memory store so I have it next session?

✻ Churned for 12s
