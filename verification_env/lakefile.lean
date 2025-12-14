import Lake
open Lake DSL

package verification_env {
  -- add package configuration options here
}

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.15.0"

@[default_target]
lean_lib VerificationEnv {
  -- add library configuration options here
}
