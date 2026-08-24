# Assemble the lifecycle manifest from the trusted task and compact agent selection.
def require($condition; $message):
  if $condition then . else error($message) end;

def exact_fields($fields):
  (keys | sort) == ($fields | sort);

# The agent may fence its object, or keep reasoning after it. Prefer the last fenced or bare
# candidate carrying exactly the selection fields; fall back to the whole text, then to a greedy span.
def selection_fields:
  ["scores", "maintained_model_ids", "obsolete_models", "new_onboarding_models", "onboarding_deployments"];

def parse_selection:
  (
    [$selection, ($selection | splits("```(?:json)?"))]
    | map(fromjson?)
    | map(select(type == "object" and exact_fields(selection_fields)))
    | last
  )
  // ($selection | fromjson?)
  // ($selection | capture("(?s)(?<json>\\{.*\\})").json | fromjson);

def model_ids($items):
  [$items[]?.model_id];

.
| require(
    type == "object" and exact_fields(["schema_version", "maintained_count", "recipe_batches"]);
    "Discovery task has an invalid shape"
  )
| require(.schema_version == 1; "Unsupported discovery task schema version")
| require(
    (.maintained_count | type) == "number" and .maintained_count >= 1 and .maintained_count == (.maintained_count | floor);
    "Discovery task maintained_count must be a positive integer"
  )
| require(
    (.recipe_batches | type) == "array" and (.recipe_batches | length) > 0
      and all(.recipe_batches[]; (type == "array" and length > 0));
    "Discovery task recipe_batches must contain non-empty arrays"
  )
| . as $task
| [$task.recipe_batches[][]] as $recipes
| parse_selection as $choice
| require(
    ($choice | type) == "object"
      and ($choice | exact_fields(selection_fields));
    "Discovery selection has an invalid shape"
  )
| require(
    ($choice.scores | type) == "array"
      and all($choice.scores[]; type == "object" and exact_fields(["model_id", "rationale", "heat"]));
    "Each score must contain exactly model_id, rationale, and heat"
  )
| ($recipes | map(.model_id)) as $recipe_ids
| ($choice.scores | model_ids(.)) as $score_ids
| require(
    ($recipe_ids | length) == ($recipe_ids | unique | length);
    "Discovery task contains duplicate model IDs"
  )
| require(
    ($score_ids | length) == ($score_ids | unique | length) and ($score_ids | sort) == ($recipe_ids | sort);
    "Scores must cover every exact recipe ID once"
  )
| ($choice.scores | INDEX(.model_id)) as $scores
| require(
    ($choice.maintained_model_ids | type) == "array"
      and all($choice.maintained_model_ids[]; type == "string")
      and ($choice.maintained_model_ids | length) == $task.maintained_count
      and ($choice.maintained_model_ids | length) == ($choice.maintained_model_ids | unique | length);
    "maintained_model_ids must contain the exact requested number of unique IDs"
  )
| [$recipes[] | select(.runnable and ((.tags | index("onboarding")) == null)) | .model_id] as $maintainable_ids
| require(
    all($choice.maintained_model_ids[]; . as $model_id | $maintainable_ids | index($model_id));
    "Maintained selections must be runnable complete recipes"
  )
| require(
    ($choice.obsolete_models | type) == "array"
      and all(
        $choice.obsolete_models[];
        type == "object" and (exact_fields(["model_id"]) or exact_fields(["model_id", "replacement_model_id"]))
      );
    "Obsolete selections have an invalid shape"
  )
| ($choice.obsolete_models | model_ids(.)) as $obsolete_ids
| [$recipes[] | select((.tags | index("onboarding")) == null) | .model_id] as $complete_ids
| require(
    ($obsolete_ids | length) == ($obsolete_ids | unique | length)
      and all($obsolete_ids[]; . as $model_id | $complete_ids | index($model_id));
    "Obsolete selections must use unique complete recipe IDs"
  )
| require(
    all(
      $choice.obsolete_models[] | select(has("replacement_model_id")) | .replacement_model_id;
      . as $model_id | $complete_ids | index($model_id)
    );
    "Obsolete replacements must use complete recipe IDs"
  )
| require(
    all($choice.maintained_model_ids[]; . as $model_id | ($obsolete_ids | index($model_id)) == null);
    "Maintained recipes cannot also be obsolete"
  )
| require(
    ($choice.new_onboarding_models | type) == "array";
    "new_onboarding_models must be an array"
  )
| require(
    ($choice.onboarding_deployments | type) == "array"
      and all($choice.onboarding_deployments[]; type == "object" and exact_fields(["model_id", "deployments"]));
    "Each sized deployment must contain exactly model_id and deployments"
  )
| (
    [$choice.onboarding_deployments[] | select((.deployments | type) == "array" and (.deployments | length) > 0)]
    | INDEX(.model_id)
  ) as $sized
| ($choice.maintained_model_ids | INDEX(.)) as $maintained
| ($obsolete_ids | INDEX(.)) as $obsolete
| {
    maintained_models: [$choice.maintained_model_ids[] | $scores[.]],
    best_effort_models: [
      $recipes[]
      | select(
          ((.tags | index("onboarding")) == null)
            and ($maintained[.model_id] == null)
            and ($obsolete[.model_id] == null)
        )
      | $scores[.model_id]
    ],
    obsolete_models: [
      $choice.obsolete_models[]
      | $scores[.model_id] + .
    ],
    onboarding_models: (
      [
        $recipes[]
        | select(.tags | index("onboarding"))
        | {
            model_id,
            task,
            rationale: $scores[.model_id].rationale,
            heat: $scores[.model_id].heat,
            deployments: (
              $sized[.model_id].deployments
                // [.deployments[] | {"deploy.gpu": .gpu, "deploy.gpu_count": .gpu_count}]
            )
          }
      ]
      + [
          $choice.new_onboarding_models[]
          | select((.model_id as $model_id | $recipe_ids | index($model_id)) == null)
          | select($sized[.model_id] != null)
          | . + {deployments: $sized[.model_id].deployments}
        ]
    )
  }
