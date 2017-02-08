## CLBNSL

The Software Package for [Curriculum Learning of Bayesian Network Structures](http://www.jmlr.org/proceedings/papers/v45/Zhao15a.pdf).

It implements **CL** algorithm proposed in [[Zhao, Yanpeng, et al.]](http://www.jmlr.org/proceedings/papers/v45/Zhao15a.pdf). CL algorithm learns Bayesian network structures from data taking advantage of the idea of *Curriculum Learning*.

If you use it in a scientific publication, we would appreciate citations to the following paper:

```
@inproceedings{zhao2015curriculum,
  title={Curriculum Learning of Bayesian Network Structures},
  author={Zhao, Yanpeng and Chen, Yetian and Tu, Kewei and Tian, Jin},
  booktitle={Proceedings of The 7th Asian Conference on Machine Learning},
  pages={269--284},
  year={2015}
}
```

### Bayesian Network Benchmarks

We use Bayesian networks on [bnlearn](http://www.bnlearn.com/bnrepository/) as benchmarks. To facilitate the programming, we parse Bayesian networks in the the format of BIF and represent the discrete **String** values with **Integer** indexes. For example, assume variable *x* takes {"rain", "snow", "wind"}, it would be {0, 1, 2} corresponding to the string value respectively in our representation.

### Synthesize Data

We sample from the distribution associated with the benchmark. To use our sampler `SampleGenerator`, you should prepare the following directories.

```
.../net/< name >/< name >.bif
```

where the ellipsis represents the *root* directory, *name* can be any benchmark name. You can reinitialize variable `AUtils.nameOfDS` to customize your benchmarks, and customize data size with `AUtils.SCALES` that will further be multiplied by 100. The sampler will generate all the necessary data files and backup them in a new directory *res* under the same *root* directory. And then the experiments are suggested to be conducted in the *res* directory.

### Inputs and Outputs

[Attribute-Relation File Format](http://www.cs.waikato.ac.nz/ml/weka/arff.html) (ARFF) files are needed for inputs.

For each different *step size*, the algorithm outputs the learned Bayesian network structures in the format of adjacent list. The following is an sample of the outputs with the suffix of **_LN.TXT**.

```
37
0 0
1 1 4
2 0
...
35 2 6 34
36 2 14 35
```

where the integer in the first line is the number of variables in the benchmark. The first column in the following lines contains the indexes of the variables, and the second column contains the number of parents of the corresponding variable. Integers following the second column are the indexes of the parents of the corresponding variable. As to the sample above, there are 37 variables, the 35th variable has 2 parents with indexes 6, 34, and so on.

In the output files with the suffix of **_ST.TXT**, the last integer following '@' specifics the *step size* with which our algorithm obtain the optimal Bayesian network structure. Here is a sample:

```
-109867.32737842221 & 8.231 @ 0
```

where the first number is the *BDeu* score, the second number is the time consumed by the algorithm on processing all the *step sizes*.

### An example

See `weka.bnsl.main.CLBayesNet` for details. And you can reproduce our experimental results reported in the paper with `weka.bnsl.main.ExperimentSet`.

### License

The project is based on [WEKA](https://weka.wikispaces.com/), it follows the same license GPL as WEKA.
