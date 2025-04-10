# tSNE

This packages provides the JavaScript/TypeScript version of the t-SNE algorithm for projecting 
high-dimensional data into a lower number of dimensions. T-SNE is often used to visualize 
embeddings. At its core tSNE attempt to preserve distances between close points.

Check out [the original paper](https://lvdmaaten.github.io/publications/papers/JMLR_2008.pdf)
to learn more.

One of the main benefits of this implementation is that it doesn't require any external 
dependencies.

## Usage example

```js
import {TSNE} from "TSNE";

const embeddings: number[][] = {
    {0.123, 0.456, ...},
    {0.123, 0.456, ...},
    ...
}

const tsne = new TSNE({ nIter: 200, perplexity: 30, learningRate: 10 });
const projectedEmbeddings: number[][] = tsne.transform(embeddings);
```

You can find an MNIST visualization demo in the [demo](https://github.com/msvana/tsne/tree/main/demo) directory.

## Configuration

The TSNE constructor accepts a configuration object. All properties of this object are optional and have default values:

- `nDims` (default `2`): number of output dimensions
- `perplexity` (default `30`): Roughly equivalent to the number of neighboring points 
    to consider for distance preservation. A value between 10 and 50 is recommended. 
- `nIter` (default `100`)
- `learningRate` (default `1`): recommended values are 1, 10, or even 100


