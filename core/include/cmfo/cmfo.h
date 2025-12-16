/* CMFO Core - Main ABI Header v1.0 */
#ifndef CMFO_H
#define CMFO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


/* Version */
#define CMFO_VERSION_MAJOR 1
#define CMFO_VERSION_MINOR 0
#define CMFO_VERSION_PATCH 0

/* Opaque handles */
typedef struct cmfo_ctx_t cmfo_ctx_t;
typedef struct cmfo_state_t cmfo_state_t;

/* Version */
typedef struct {
  uint16_t major;
  uint16_t minor;
  uint16_t patch;
} cmfo_version_t;

/* 7D vector */
typedef struct {
  double v[7];
} cmfo_vec7_t;

/* Result codes */
typedef enum {
  CMFO_OK = 0,
  CMFO_ERROR_INVALID_ARG = 1,
  CMFO_ERROR_OUT_OF_MEMORY = 2,
  CMFO_ERROR_INVALID_STATE = 3,
  CMFO_ERROR_LICENSE = 4,
  CMFO_ERROR_SECURITY = 5,
  CMFO_ERROR_UNKNOWN = 99
} cmfo_result_t;

/* Configuration modes */
typedef enum {
  CMFO_MODE_STUDY = 0x01,
  CMFO_MODE_RESEARCH = 0x02,
  CMFO_MODE_ENTERPRISE = 0x04,
  CMFO_MODE_DETERMINISTIC = 0x10,
  CMFO_MODE_AUDIT = 0x20
} cmfo_mode_t;

/* Configuration */
typedef struct {
  cmfo_mode_t mode;
  uint64_t memory_limit_bytes;
  const char *license_key;
  const char *audit_log_path;
  uint32_t flags;
} cmfo_config_t;

/* Core functions */
cmfo_version_t cmfo_get_version(void);
cmfo_ctx_t *cmfo_init(const cmfo_config_t *config);
void cmfo_destroy(cmfo_ctx_t *ctx);
const char *cmfo_get_error(cmfo_ctx_t *ctx);

/* State management */
cmfo_state_t *cmfo_state_create(cmfo_ctx_t *ctx, const cmfo_vec7_t *vec);
cmfo_result_t cmfo_state_get_vec(const cmfo_state_t *state, cmfo_vec7_t *vec);
void cmfo_state_destroy(cmfo_state_t *state);

/* Automaton */
cmfo_state_t *cmfo_step(cmfo_ctx_t *ctx, const cmfo_state_t *state);
cmfo_state_t *cmfo_step_reverse(cmfo_ctx_t *ctx, const cmfo_state_t *state);
cmfo_state_t *cmfo_evolve(cmfo_ctx_t *ctx, const cmfo_state_t *state,
                          int64_t n);

/* Algebra */
cmfo_result_t cmfo_compose(cmfo_ctx_t *ctx, const cmfo_vec7_t *v,
                           const cmfo_vec7_t *w, cmfo_vec7_t *result);
cmfo_result_t cmfo_modulate(cmfo_ctx_t *ctx, double scalar,
                            const cmfo_vec7_t *v, cmfo_vec7_t *result);
cmfo_result_t cmfo_negate(cmfo_ctx_t *ctx, const cmfo_vec7_t *v,
                          cmfo_vec7_t *result);
cmfo_result_t cmfo_distance(cmfo_ctx_t *ctx, const cmfo_vec7_t *v,
                            const cmfo_vec7_t *w, double *distance);

/* Language */
cmfo_result_t cmfo_parse(cmfo_ctx_t *ctx, const char *text, cmfo_vec7_t *vec);
cmfo_result_t cmfo_solve(cmfo_ctx_t *ctx, const char *equation,
                         char **solution);

/* Memory */
cmfo_result_t cmfo_memory_store(cmfo_ctx_t *ctx, const cmfo_state_t *state,
                                char **id);
cmfo_state_t *cmfo_memory_load(cmfo_ctx_t *ctx, const char *id);

/* Audit */
cmfo_result_t cmfo_audit_get(cmfo_ctx_t *ctx, uint64_t index, char **entry);
cmfo_result_t cmfo_audit_verify(cmfo_ctx_t *ctx, bool *valid);

#ifdef __cplusplus
}
#endif

#endif /* CMFO_H */
