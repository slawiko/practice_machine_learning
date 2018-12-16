./svm-scale -s ../data/gisette/scale -l 0 ../data/gisette/gisette.scale.data > ../data/gisette/gisette.scale.scaled
./svm-scale -r ../data/gisette/scale -l 0 ../data/gisette/gisette.test.data > ../data/gisette/gisette.test.scaled
./svm-scale -r ../data/gisette/scale -l 0 ../data/gisette/gisette.train.data > ../data/gisette/gisette.train.scaled
