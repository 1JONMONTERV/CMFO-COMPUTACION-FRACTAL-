/* CMFO Core - Context Management Implementation */
#include "cmfo/cmfo.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>


#define PHI 1.618033988749895

/* Internal context structure */
struct cmfo_ctx_t {
  cmfo_config_t config;
  char error_msg[256];
  double lambda[7]; /* Fractal weights */
};

/* Internal state structure */
struct cmfo_state_t {
  cmfo_vec7_t vec;
};

/* Version */
cmfo_version_t cmfo_get_version(void) {
  cmfo_version_t ver = {CMFO_VERSION_MAJOR, CMFO_VERSION_MINOR,
                        CMFO_VERSION_PATCH};
  return ver;
}

/* Initialize context */
cmfo_ctx_t *cmfo_init(const cmfo_config_t *config) {
  cmfo_ctx_t *ctx = (cmfo_ctx_t *)malloc(sizeof(cmfo_ctx_t));
  if (!ctx)
    return NULL;

  /* Set default config */
  if (config) {
    ctx->config = *config;
  } else {
    ctx->config.mode = CMFO_MODE_STUDY;
    ctx->config.memory_limit_bytes = 0;
    ctx->config.license_key = NULL;
    ctx->config.audit_log_path = NULL;
    ctx->config.flags = 0;
  }

  /* Initialize fractal weights: λᵢ = φ^(i-1) */
  double phi_power = 1.0;
  for (int i = 0; i < 7; i++) {
    ctx->lambda[i] = phi_power;
    phi_power *= PHI;
  }

  ctx->error_msg[0] = '\0';
  return ctx;
}

/* Destroy context */
void cmfo_destroy(cmfo_ctx_t *ctx) {
  if (ctx) {
    free(ctx);
  }
}

/* Get error message */
const char *cmfo_get_error(cmfo_ctx_t *ctx) {
  return ctx ? ctx->error_msg : "Invalid context";
}

/* Set error message */
static void set_error(cmfo_ctx_t *ctx, const char *msg) {
  if (ctx && msg) {
    strncpy(ctx->error_msg, msg, sizeof(ctx->error_msg) - 1);
    ctx->error_msg[sizeof(ctx->error_msg) - 1] = '\0';
  }
}

/* Create state from vector */
cmfo_state_t *cmfo_state_create(cmfo_ctx_t *ctx, const cmfo_vec7_t *vec) {
  if (!ctx || !vec) {
    if (ctx)
      set_error(ctx, "Invalid arguments");
    return NULL;
  }

  cmfo_state_t *state = (cmfo_state_t *)malloc(sizeof(cmfo_state_t));
  if (!state) {
    set_error(ctx, "Out of memory");
    return NULL;
  }

  state->vec = *vec;
  return state;
}

/* Get vector from state */
cmfo_result_t cmfo_state_get_vec(const cmfo_state_t *state, cmfo_vec7_t *vec) {
  if (!state || !vec) {
    return CMFO_ERROR_INVALID_ARG;
  }

  *vec = state->vec;
  return CMFO_OK;
}

/* Destroy state */
void cmfo_state_destroy(cmfo_state_t *state) {
  if (state) {
    free(state);
  }
}

/* Normalize vector */
static void normalize(cmfo_vec7_t *vec) {
  double norm = 0.0;
  for (int i = 0; i < 7; i++) {
    norm += vec->v[i] * vec->v[i];
  }
  norm = sqrt(norm);

  if (norm > 1e-10) {
    for (int i = 0; i < 7; i++) {
      vec->v[i] /= norm;
    }
  }
}

/* Clip vector components to [-1, 1] */
static void clip(cmfo_vec7_t *vec) {
  for (int i = 0; i < 7; i++) {
    if (vec->v[i] > 1.0)
      vec->v[i] = 1.0;
    if (vec->v[i] < -1.0)
      vec->v[i] = -1.0;
  }
}

/* Compose: v ⊕ w */
cmfo_result_t cmfo_compose(cmfo_ctx_t *ctx, const cmfo_vec7_t *v,
                           const cmfo_vec7_t *w, cmfo_vec7_t *result) {
  if (!ctx || !v || !w || !result) {
    if (ctx)
      set_error(ctx, "Invalid arguments");
    return CMFO_ERROR_INVALID_ARG;
  }

  /* Add vectors */
  for (int i = 0; i < 7; i++) {
    result->v[i] = v->v[i] + w->v[i];
  }

  /* Normalize */
  normalize(result);

  return CMFO_OK;
}

/* Scalar modulation: a ⊗ v */
cmfo_result_t cmfo_modulate(cmfo_ctx_t *ctx, double scalar,
                            const cmfo_vec7_t *v, cmfo_vec7_t *result) {
  if (!ctx || !v || !result) {
    if (ctx)
      set_error(ctx, "Invalid arguments");
    return CMFO_ERROR_INVALID_ARG;
  }

  /* Multiply by scalar */
  for (int i = 0; i < 7; i++) {
    result->v[i] = scalar * v->v[i];
  }

  /* Clip to [-1, 1] */
  clip(result);

  return CMFO_OK;
}

/* Negation: NEG(v) */
cmfo_result_t cmfo_negate(cmfo_ctx_t *ctx, const cmfo_vec7_t *v,
                          cmfo_vec7_t *result) {
  if (!ctx || !v || !result) {
    if (ctx)
      set_error(ctx, "Invalid arguments");
    return CMFO_ERROR_INVALID_ARG;
  }

  /* Copy vector */
  *result = *v;

  /* Invert truth axis (index 1) */
  result->v[1] = -result->v[1];

  /* Normalize to preserve norm */
  normalize(result);

  return CMFO_OK;
}

/* Fractal distance: d_φ(v, w) */
cmfo_result_t cmfo_distance(cmfo_ctx_t *ctx, const cmfo_vec7_t *v,
                            const cmfo_vec7_t *w, double *distance) {
  if (!ctx || !v || !w || !distance) {
    if (ctx)
      set_error(ctx, "Invalid arguments");
    return CMFO_ERROR_INVALID_ARG;
  }

  /* d_φ(v,w) = √(Σ λᵢ(vᵢ-wᵢ)²) */
  double sum = 0.0;
  for (int i = 0; i < 7; i++) {
    double diff = v->v[i] - w->v[i];
    sum += ctx->lambda[i] * diff * diff;
  }

  *distance = sqrt(sum);
  return CMFO_OK;
}

/* Automaton step: X_{t+1} = U_φ(X_t) */
cmfo_state_t *cmfo_step(cmfo_ctx_t *ctx, const cmfo_state_t *state) {
  if (!ctx || !state) {
    if (ctx)
      set_error(ctx, "Invalid arguments");
    return NULL;
  }

  /* Simple rotation in 7D space */
  cmfo_vec7_t new_vec;
  double angle = 2.0 * M_PI / PHI; /* Golden angle */

  /* Rotate in each plane */
  for (int i = 0; i < 7; i++) {
    int j = (i + 1) % 7;
    double cos_a = cos(angle / ctx->lambda[i]);
    double sin_a = sin(angle / ctx->lambda[i]);

    new_vec.v[i] = state->vec.v[i] * cos_a - state->vec.v[j] * sin_a;
  }

  normalize(&new_vec);
  return cmfo_state_create(ctx, &new_vec);
}

/* Reverse step */
cmfo_state_t *cmfo_step_reverse(cmfo_ctx_t *ctx, const cmfo_state_t *state) {
  if (!ctx || !state) {
    if (ctx)
      set_error(ctx, "Invalid arguments");
    return NULL;
  }

  /* Reverse rotation */
  cmfo_vec7_t new_vec;
  double angle = -2.0 * M_PI / PHI;

  for (int i = 0; i < 7; i++) {
    int j = (i + 1) % 7;
    double cos_a = cos(angle / ctx->lambda[i]);
    double sin_a = sin(angle / ctx->lambda[i]);

    new_vec.v[i] = state->vec.v[i] * cos_a - state->vec.v[j] * sin_a;
  }

  normalize(&new_vec);
  return cmfo_state_create(ctx, &new_vec);
}

/* Evolve n steps */
cmfo_state_t *cmfo_evolve(cmfo_ctx_t *ctx, const cmfo_state_t *state,
                          int64_t n) {
  if (!ctx || !state) {
    if (ctx)
      set_error(ctx, "Invalid arguments");
    return NULL;
  }

  cmfo_state_t *current = cmfo_state_create(ctx, &state->vec);
  if (!current)
    return NULL;

  int64_t steps = n > 0 ? n : -n;
  for (int64_t i = 0; i < steps; i++) {
    cmfo_state_t *next =
        n > 0 ? cmfo_step(ctx, current) : cmfo_step_reverse(ctx, current);

    cmfo_state_destroy(current);
    current = next;

    if (!current)
      break;
  }

  return current;
}
