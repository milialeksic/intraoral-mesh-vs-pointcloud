# Python normally buffers output when it's piped; -u makes it print (and log) immediately
New-Item -ItemType Directory -Force -Path logs | Out-Null

foreach ($seed in 1, 2) {
    foreach ($graph in "mesh", "knn") {
        $log = "logs\train_${graph}_seed$seed.log"
        python -u src\train.py --graph $graph --seed $seed `
            --out runs\${graph}_s$seed --results-dir results\seed$seed |
            Tee-Object -FilePath $log
    }
}