# GANN reproduction package.
# Structure:
#   models/transformer.py       : vanilla Transformer building blocks (paper Sec. III-B)
#   models/global_attention.py  : Global-Attention (GANN) cross-attention (paper Sec. III-C, Eq.9-14)
#   models/gann.py              : full captioner = encoder(CLS token + label head) + decoder(GANN) + losses
#   dataloader.py               : pre-extracted bottom-up features + Karpathy captions
#   cider_reward.py             : CIDEr-D reward for SCST (paper Sec. III-D, Eq.16-17)
#   language_eval.py            : COCO evaluation wrapper (BLEU/METEOR/ROUGE-L/CIDEr-D/SPICE)
#   eval_utils.py               : greedy / beam decoding + evaluation loop
__version__ = '0.1.0'
