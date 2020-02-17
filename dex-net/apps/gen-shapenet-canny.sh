
unset PYTHONPATH
python3 generate-shapenet-dataset-canny.py &> gen-shapenet-canny.log &
echo "Gen launched. Showing log at gen-shapenet-canny.log (CTRL-C to abort watching the log)"
sleep 2
tail -f gen-shapenet-canny.log
