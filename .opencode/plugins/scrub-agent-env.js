const secretNames = [
  "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
  "ACTIONS_ID_TOKEN_REQUEST_URL",
  "ACTIONS_RUNTIME_TOKEN",
  "AGENT_KEY_FILE",
  "APP_KEY_FILE",
  "CLOUDRIFT_API_KEY",
  "CLOUDRIFT_API_URL",
  "EXPERIMENT_APP_PRIVATE_KEY",
  "GCP_CONFIG_DIR",
  "GCP_KEY_FILE",
  "GCP_SERVICE_ACCOUNT",
  "GH_TOKEN",
  "GITHUB_TOKEN",
  "GOOGLE_APPLICATION_CREDENTIALS",
]

export const ScrubAgentEnvironment = async () => ({
  "shell.env": async (_input, output) => {
    for (const name of secretNames) delete output.env[name]
  },
})
