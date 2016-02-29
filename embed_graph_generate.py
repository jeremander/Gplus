import subprocess

embeddings = ['adj', 'adj+diag', 'normlap', 'regnormlap']
ks = [50, 100, 200, 400]

for k in ks:
    for embedding in embeddings:
        subcmd = "python3 embed_graph.py -e %s -k %d" % (embedding, k)
        cmd = 'qsub -q all.q -l num_proc=1,mem_free=%dG,h_rt=120:00:00 -b Y -V -cwd -j yes -o . -N %s_k%d "%s"' % (k // 15 + 1, embedding, k, subcmd)
        print(cmd)
        subprocess.Popen(cmd, shell = True, stdout = subprocess.PIPE)