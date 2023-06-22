# compute-embeddings
A small tool to compute the embeddings of a list of JSON documents

## Installation

You must have the Rust toolchain installed, it is pretty easy to install [by following the official tutorial](https://www.rust-lang.org/tools/install).

## Example Input Documents

```json
[
  {
    "name": "3-Year Unlimited Cloud Storage Service Activation Card - Other",
    "description": "Enjoy 3 years of unlimited Cloud storage service with this activation card, which allows you to remotely access your favorite music, movies and other media via a compatible device and enables private file sharing with loved ones.",
    "brand": "Pogoplug",
    "categories": [
      "Best Buy Gift Cards",
      "Entertainment Gift Cards"
    ],
    "hierarchicalCategories": {
      "lvl0": "Best Buy Gift Cards",
      "lvl1": "Best Buy Gift Cards > Entertainment Gift Cards"
    },
    "type": "Online data backup",
    "price": 69,
    "price_range": "50 - 100",
    "image": "https://cdn-demo.algolia.com/bestbuy/1696302_sc.jpg",
    "url": "http://www.bestbuy.com/site/3-year-unlimited-cloud-storage-service-activation-card-other/1696302.p?id=1219066776306&skuId=1696302&cmp=RMX&ky=1uWSHMdQqBeVJB9cXgEke60s5EjfS6M1W",
    "free_shipping": true,
    "popularity": 10000,
    "rating": 2,
    "objectID": "1696302"
  }
]
```

## Usage for Meilisearch

```bash
cat file.json | cargo run --release --bin ce-dataset -- --batched-documents 8 --semantic-api all-mini-lm-l6v2 --documents-style meilisearch name description brand categories _type object_id > file-with-embeddings.json
```

### Example Output File

```json5
[
  {
    "name": "3-Year Unlimited Cloud Storage Service Activation Card - Other",
    "description": "Enjoy 3 years of unlimited Cloud storage service with this activation card, which allows you to remotely access your favorite music, movies and other media via a compatible device and enables private file sharing with loved ones.",
    "brand": "Pogoplug",
    "categories": [
      "Best Buy Gift Cards",
      "Entertainment Gift Cards"
    ],
    "hierarchicalCategories": {
      "lvl0": "Best Buy Gift Cards",
      "lvl1": "Best Buy Gift Cards > Entertainment Gift Cards"
    },
    "type": "Online data backup",
    "price": 69.0,
    "price_range": "50 - 100",
    "image": "https://cdn-demo.algolia.com/bestbuy/1696302_sc.jpg",
    "url": "http://www.bestbuy.com/site/3-year-unlimited-cloud-storage-service-activation-card-other/1696302.p?id=1219066776306&skuId=1696302&cmp=RMX&ky=1uWSHMdQqBeVJB9cXgEke60s5EjfS6M1W",
    "free_shipping": true,
    "popularity": 10000,
    "rating": 2,
    "objectID": "1696302",
    "_vectors": [
      -0.10141887,
      0.009569897,
      0.04121973
      // [...]
    ]
  }
]
```

### Query Meilisearch

Once you have sent your documents into your Meilisearch index you can query it with vectors too!

```bash
echo '{
    "vector": '$(cargo run --release --bin ce-query -- --semantic-api all-mini-lm-l6v2 'toys r us')',
    "attributesToRetrieve": ["name", "description"]
}' \
| curl -X POST -H 'content-type: application/json' 'localhost:7700/indexes/movies/search' --data-binary @- | jq
```
