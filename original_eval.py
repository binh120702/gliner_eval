import argparse

from gliner import GLiNER
from gliner.evaluation.evaluate import get_for_all_path, compare_decoding_algorithms
import os

def create_parser():
    parser = argparse.ArgumentParser(description="Span-based NER")
    parser.add_argument("--model", type=str, default="logs/model_12000", help="Path to model folder")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to model folder")
    parser.add_argument('--data', type=str, default='data/NER/', help='Path to the eval datasets directory')
    parser.add_argument('--compare', action='store_true', help='Run and compare greedy/NMS/MWIS decoding')
    parser.add_argument('--nms_iou', type=float, default=0.5, help='IoU threshold for NMS decoding')
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    model = GLiNER.from_pretrained(args.model, load_tokenizer=True).to("cuda:0")
    if args.compare:
        compare_decoding_algorithms(model, -1, args.log_dir, args.data, iou_threshold=args.nms_iou)
    else:
        get_for_all_path(model, -1, args.log_dir, args.data)