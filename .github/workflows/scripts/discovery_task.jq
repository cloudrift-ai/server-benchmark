# Convert expanded recipe-query rows into bounded discovery batches.
if .schema_version != 1 then
  error("Unsupported recipe query schema version")
elif ($maintained_count | type) != "number" or $maintained_count < 1 or $maintained_count != ($maintained_count | floor) then
  error("maintained_count must be a positive integer")
elif ($batch_size | type) != "number" or $batch_size < 1 or $batch_size != ($batch_size | floor) then
  error("batch_size must be a positive integer")
else
  [
    .rows
    | group_by(.model_id)[]
    | . as $rows
    | $rows[0]
    | {
        name,
        path: .recipe_path,
        model_id,
        tags,
        task,
        runnable,
        rationale,
        heat,
        deployments: [
          $rows[].deployment
          | select(. != null)
          | {gpu, gpu_count, context_length}
        ]
      }
  ] as $recipes
  | ($recipes | map(select(.runnable and ((.tags | index("onboarding")) == null))) | length) as $maintainable_count
  | if ($recipes | length) == 0 then
      error("Recipe query returned no recipes")
    elif $maintained_count > $maintainable_count then
      error("maintained_count exceeds the runnable complete recipe count")
    else
      {
        schema_version: 1,
        maintained_count: $maintained_count,
        recipe_batches: [
          range(0; ($recipes | length); $batch_size) as $index
          | $recipes[$index:$index + $batch_size]
        ]
      }
    end
end
