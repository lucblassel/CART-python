# Steps in the CART decision tree inference algorithm

## Nomenclature
We consider:
$$
Y \rightarrow target\\
X_{\{1,\cdots,p\}} \rightarrow explanatory\ features
$$

For a classification tree $Y_i \in \{1,2,\cdots,k\}$, where $k$ is the number of possible classes.  
On the other hand for a regression tree $Y_i \in \mathbb{R}$

----

## splitting the data
splitting on a feature. -> 2 different cases:
 - feature is numerical
 - feature is categorical
(there is ordered categorical which is basically the same as a number so whatever)

### A. numerical feature
For a given node there are $n$ data points. Let us consider the numerical expalatory feature $X_{num}$. In this case we have a maximum number of splits $n-1$ corresponding to all the splits:

$$ S = \{ X_{num} \leq x_i\},\{X_{num} > x_i\}\quad i \in \{1,\cdots,n-1\}$$
We stop at $n-1$ because, for the maximum value of $X_{num}$ all points are smaller or equal to it, meaning we send all of our data points to one side of the split and leave the other side empty. This of course is not split, so we don't count that possibility. 

### B. categorical feature
Let us consider now the categorical feature $X_{cat}$, which has $k$ possible levels. the possible number of splits is $2^{k-1} - 1$. A given split can be defined as:

$$ S = \{X_{cat} \in \mathcal{A}\},\{X_{cat} \in \mathcal{\overline{A}}\}$$
With $\mathcal{A}$ a subset of the possible levels, meaning:  $\mathcal{A} \subseteq \{1,\cdots,k\}$ , and $\mathcal{\overline{A}}$ it's opposite.


_How many of these subsets are there?_  
Let's consider first the case of a feature called $C$ with 3 levels:
$$C \in \{red,\ blue,\ green\}$$
To create a subset we must choose which values are in or not the subset. For example the subset $\mathcal{A} = \{red,\ green\}$ what we are saying is $red\in \mathcal{A}$ and $green\in \mathcal{A}$ and $blue\notin \mathcal{A}$. So each subset is a set of 3 values indicating presence or absence of a given level of $C$ in the subset. With this we can easily calculate the number of possible subsets.  
For the first value (presence or absence of $red$ in the subset) there are $2$ possible options, for the second value we also have $2$ potentail values and the same for the third values.  
So our total number of possible subsets is: 
$$ N_{sets}=2\times2\times2 = 2^3$$
And if we have $k$ possible levels, our number of possible subsets becomes:
$$N_{sets}=2^k$$


_but further up it was_ $2^{k-1} -1$ _why?_  
Well this comes from the fact that we are looking for splits, not just for subsets. And splits are symmetrical. For our example above, if we have: 

$$ 
\mathcal{A} = \{red, green\} \Leftrightarrow \mathcal{\overline{A}} = \{blue\} \\
\quad\\
S_1 = \{C\in\mathcal{A}\},\{C\in\mathcal{\overline{A}}\}\\
S_2 = \{C\in\mathcal{\overline{A}}\},\{C\in\mathcal{A}\}\\
$$

It is easy to see that $S_1 = S_2$, that symmetry is why on out number of possible splits we have $2^{k-1}$ instead of simply $2^k$. The explanation for why it is $2^{k-1}-1$ and not $2^{k-1}$ is the same as for numerical features, because if $\mathcal{A}=\{red,\ blue,\ green\}$ then Our splits has all the points one one side and and empty set on the other, so it is not a split, so we remove that possibility and that's how we end up with $2^{k-1}-1$ .  

So now we have all of the possible splits in our data, but how do we choose the best one?

## Choosing the best split

Since we want to use the tree to predict either a class or a value, we want the leafs of the tree to be as "pure" as possible, meaning we want the examples in each leaf to be similar. To get that we need a way to measure the "purity" of a node, so how similar all data points are in that node.  
Therefore, to chose the best split, we choose the one that maximizes this "purity" measure in the child nodes. In practice we don't measure "purity" but rather "impurity". There are several of these measures. In this whole section let's consider a node $t$, and for an impurity measure $i$ we can define $i(t)$ as the impurity of this node.

### The Gini index
The Gini index is a way to measure impurity in classification trees. Let $p_i$ be the probability of having class $i$ in our node, and $k$ the number of classes in the node. The gini index $G$ is:
$$
G(t) = 1 - \sum^k_{i=1} p_i^2
$$

Since we don't know the real probability $p_i$ for a given class, we just approximate it by the frequency of said class in the node:
$$
p_i = \frac{n_i}{n}
$$
with $n_i$ the number of examples of class $i$ in the considered node and $n$ the total number of examples in said node. 

### Entropy
Entropy is also an impurity measure that is commonly used in CART with classification trees. If we use the same notation as for the Gini index we can define the Entropy of a node, $E$, as:

$$
E(t) = \sum^k_{i=1} p_i \log(p_i)
$$

usually the base $2$ logarithm is used.

### RSS
The Residual Sum of Squares (RSS) is used in regression trees, where examples have an outcome value instead of an outcome class. For a given node, let $n$ be the number of examples at that node, $y_i$ the outcome value of the $i^{th}$ example, and $\mu$ the mean outcome values of all examples of the node. We can define :

$$
RSS(t) = \sum^n_{i=1} (y_i - \mu)^2
$$
  
  

To then choose the optimal split we choose the one that leads to the maximal decrease in impurity. So if we are in a node $t$ and a given split defines nodes $L$ and $R$ the ***left*** and ***right*** child nodes of $t$ respectively. We can define the decrease in impurity $\Delta i$ as:
$$
\Delta i = i(t) - p_L\cdot i(L) - p_R\cdot i(R)
$$

with $i(L)$ and $i(R)$ the impurities of the child nodes and $p_L$ and $p_R$ the probabilities of a sample from node $t$ will go to node $L$ or node $R$, again we equate this to the proportions of cases that go to nodes $R$ and $L$ so $p_L = \frac{n_L}{n_t}$ (with $n_L$ and $n_t$ the number of cases in nodes $L$ and $t$)

## The algorithm

The basic algorithm is actually quite simple: 

1. Find all possible splits in dataset
2. calculate decrease in impurity for each of these splits 
3. Choose split for which decrease in impurity is maximal
4. on each half of the split, start this process again


We can implement it in a recursive manner thusly:


```` 
function infer_tree(dataset) {

    #stopping condition
    if dataset_is_pure(dataset){
        stop
    }

    splits = find_all_possible_splits(dataset)
    impurities = calculate_impurity_decrease(splits)

    # choose split with maximum decrease
    best_split = splits[max(impurities)] 
    
    left_dataset = best_split[0]
    right_dataset = best_split[1]

    #recursive part
    infer_tree(left_dataset)
    infer_tree(right_dataset)

}
````
This is the simplest form of the algorithm, it results in a tree that grows until all leaves are pure (meaning that they contain examples of only one class), in most cases this is too stringent and can result in overfitting. To avoid this several strategies can be used:

* you can define a minimum number of examples in each leaf, and if the dataset at a node is smaller or equal than that minimum then the node is not split and stays a leaf. 
* you can prune the tree and collapse leaves together.


