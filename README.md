# PackHero: A Scalable Graph-based Approach for Efficient Packer Identification

This repo contains the code used to implement PackHero and perform all the experiments useful for our research around this novel approach.

The idea behind PackHero is to use graphs to represent packed PEs and search for similarities among a set of well-analyzed graphs of packed PEs labeled with the known Packer. Through these similarities, the Packer that has packed the input PE is then identified.

## Install Environment
The tool execution and the experiment replication need [Anaconda](https://www.anaconda.com/). Once you have Anaconda installed, import the conda environment:
```console
conda env create -f environment.yml
```
To run PackHero and experiments, in particular for the graph extraction part, you need to have installed [radare2](https://rada.re/n/).
## Use PackHero
To use PackHero you have to run `packhero.sh`, make it executable and run it with the following settings:
```console
./packhero.sh --file|--dir --mean|--majority|--clustering <filepath|directorypath> [--discard]
```

### Flags

- `--file` - Use this flag when specifying a single file for analysis.
- `--dir` - Use this flag when specifying a directory, to analyze all files within that directory.
- `<filepath|directorypath>` - Specify the path to the file or directory you want to analyze. The file(s) can be both PEs and already extracted graphs saved in graphml format.
- `--mean` (optional) - Run the tool in 'mean' mode. Use this flag to process the data using the mean method.
- `--majority` (optional) - Run the tool in 'majority' mode. Use this flag to process the data using the majority method.
- `--clustering` (optional) - Run the tool in 'clustering' mode. Use this flag to process the data using the clustering method.
- `--discard` (optional) - Use this flag to discard dirty graphs.


