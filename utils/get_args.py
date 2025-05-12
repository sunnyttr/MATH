import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--k-bits", type=int, default=16, help="length of hash codes.")
    parser.add_argument("--res-mlp-layers", type=int, default=2, help="the number of Residual MLP blocks.")
    parser.add_argument("--transformer-layers", type=int, default=2, help="the number of Transformer Encoder layers.")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument("--valid-freq", type=int, default=1, help="To valid every $valid-freq$ epochs.")
    parser.add_argument("--rank", type=int, default=0, help="GPU rank")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--clip-lr", type=float, default=0.000002, help="learning rate for CLIP in MATH.")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for other modules in MATH.")

    parser.add_argument("--is-train", action="store_true")
    
    # hyper parameters
    parser.add_argument("--hyper-intra", type=float, default=4, help="weight of the intra-modal similarity preservation loss, which is labeled as $\ alpha$ in paper.")
    parser.add_argument("--hyper-inter", type=float, default=13, help="weight of the inter-modal similarity preservation loss, which is labeled as $\ beta$ in paper.")  
    parser.add_argument("--hyper-MSE", type=float, default=5, help="weight of the global-local MSE loss, which is labeled as $\ delta$ in paper.")
    parser.add_argument("--hyper-quan", type=float, default=7, help="weight of the quantization loss, which is labeled as $\ gamma$ in paper.")
    parser.add_argument("--hyper-contrast", type=float, default=57, help="weight of the multi-label cross-modal contrastive alignment loss, which is labeled as $\ lambda$ in paper.")

    # other
    parser.add_argument("--clip-path", type=str, default="./cache/ViT-B-32.pt", help="pretrained clip path.")
    parser.add_argument("--dataset", type=str, default="flickr25k", help="choose from [flickr25k, nuswide, coco]")
    parser.add_argument("--query-num", type=int, default=2000)
    parser.add_argument("--train-num", type=int, default=10000)

    parser.add_argument("--pretrained", type=str, default="", help="pretrained model path.")
    parser.add_argument("--index-file", type=str, default="index.mat")
    parser.add_argument("--caption-file", type=str, default="caption.mat")
    parser.add_argument("--label-file", type=str, default="label.mat")
    parser.add_argument("--label_t-file", type=str, default="label_t.mat")
    parser.add_argument("--max-words", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--result-name", type=str, default="RESULT_MATH_FLICKR", help="result dir name.")

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-proportion", type=float, default=0.05, help="Proportion of training to perform learning rate warmup.")

    args = parser.parse_args()

    import datetime
    _time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not args.is_train:
        _time += "_test"
    k_bits = args.k_bits
    parser.add_argument("--save-dir", type=str, default=f"./{args.result_name}/{args.dataset}_{k_bits}/{_time}")
    args = parser.parse_args()

    return args
