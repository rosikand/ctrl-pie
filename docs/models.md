---
title: "Models"
description: "Browse namespace model repositories and prepare revision-pinned policies for inference."
icon: "package"
---

The **Models** subview under Training discovers model repositories in the exact
configured Hugging Face namespace. It complements Trainer API run metadata:
the Hub remains the source of truth for cards, weights, checkpoints, and Git
revisions.

## Browse model artifacts

Each model card shows available:

- repository ID, visibility, gated state, and immutable current revision;
- description, pipeline tag, library name, and tags;
- base-model and dataset references from the model card;
- checkpoint-like artifact paths discovered at that revision; and
- last-modified time with links pinned to the same commit.

Choose **Refresh** to bypass the 30-second private enumeration and card caches.
A missing or malformed card degrades one model rather than failing the entire
namespace.

## Revisions and checkpoint paths

The model's `revision` is a Hub Git commit and is suitable for inference.
Entries such as `checkpoints/001000/model.safetensors` are files inside that
revision. They are useful metadata, but they are not branches, tags, or commit
SHAs and cannot be sent as `checkpoint_revision`.

The Inference page therefore offers one deployable Git revision per model and
shows checkpoint artifact paths as read-only context.

## Deployable LeRobot layout

A real LeRobot policy must be loadable from the selected model repository
root. At minimum, that immutable revision needs root-level policy assets such
as:

```text
config.json
model.safetensors
policy_preprocessor.json
policy_postprocessor.json
```

Uploading only below `checkpoints/<step>/pretrained_model` can make a model
visible in the catalog while leaving it undeployable. Upload the contents of
the final `pretrained_model` directory to the model repository root, then
register the returned commit SHA with ctrl-π.

The executable example in [Trainer API](/trainer-api#end-to-end-lerobot-fine-tune-and-checkpoint-flow)
creates a new private model repository, uploads exactly that root layout, and
registers the immutable revision.

## Inference identity

Real deployment accepts only a model owned by `HF_NAMESPACE`. The backend
resolves a requested ref to a lowercase 40-character commit before creating a
Modal resource. That repository and SHA are persisted, baked into an identity
marker, and checked again at endpoint start and before robot motion.

The Modal image receives `HF_TOKEN` only while downloading the selected
revision. Serving then runs offline so nested model or processor loads cannot
resolve a mutable dependency.

Continue to [Inference](/inference) for supported runtimes, GPU sizes, and the
deploy/start/stop lifecycle.
