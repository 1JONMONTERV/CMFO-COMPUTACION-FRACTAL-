/* CMFO Core - Basic Test */
#include "cmfo/cmfo.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>


#define TEST(name) printf("Testing %s...\n", name)
#define PASS() printf("  PASS\n")

int main() {
  printf("CMFO Core C - Basic Tests\n");
  printf("==========================\n\n");

  /* Test 1: Version */
  TEST("version");
  cmfo_version_t ver = cmfo_get_version();
  assert(ver.major == 1);
  assert(ver.minor == 0);
  assert(ver.patch == 0);
  PASS();

  /* Test 2: Context initialization */
  TEST("context init/destroy");
  cmfo_ctx_t *ctx = cmfo_init(NULL);
  assert(ctx != NULL);
  cmfo_destroy(ctx);
  PASS();

  /* Test 3: State creation */
  TEST("state create/destroy");
  ctx = cmfo_init(NULL);
  cmfo_vec7_t vec = {{1, 0, 0, 0, 0, 0, 0}};
  cmfo_state_t *state = cmfo_state_create(ctx, &vec);
  assert(state != NULL);

  cmfo_vec7_t retrieved;
  cmfo_result_t result = cmfo_state_get_vec(state, &retrieved);
  assert(result == CMFO_OK);
  assert(fabs(retrieved.v[0] - 1.0) < 1e-10);

  cmfo_state_destroy(state);
  cmfo_destroy(ctx);
  PASS();

  /* Test 4: Compose */
  TEST("compose");
  ctx = cmfo_init(NULL);
  cmfo_vec7_t v1 = {{1, 0, 0, 0, 0, 0, 0}};
  cmfo_vec7_t v2 = {{0, 1, 0, 0, 0, 0, 0}};
  cmfo_vec7_t composed;

  result = cmfo_compose(ctx, &v1, &v2, &composed);
  assert(result == CMFO_OK);

  /* Result should be normalized */
  double norm = 0;
  for (int i = 0; i < 7; i++) {
    norm += composed.v[i] * composed.v[i];
  }
  assert(fabs(sqrt(norm) - 1.0) < 1e-6);

  cmfo_destroy(ctx);
  PASS();

  /* Test 5: Distance */
  TEST("distance");
  ctx = cmfo_init(NULL);
  double dist;
  result = cmfo_distance(ctx, &v1, &v2, &dist);
  assert(result == CMFO_OK);
  assert(dist > 0);

  cmfo_destroy(ctx);
  PASS();

  /* Test 6: Negate */
  TEST("negate");
  ctx = cmfo_init(NULL);
  cmfo_vec7_t v = {{0, 1, 0, 0, 0, 0, 0}};
  cmfo_vec7_t negated;

  result = cmfo_negate(ctx, &v, &negated);
  assert(result == CMFO_OK);
  assert(negated.v[1] < 0); /* Truth axis inverted */

  cmfo_destroy(ctx);
  PASS();

  /* Test 7: Automaton step */
  TEST("automaton step");
  ctx = cmfo_init(NULL);
  state = cmfo_state_create(ctx, &v1);
  cmfo_state_t *next = cmfo_step(ctx, state);
  assert(next != NULL);

  cmfo_state_destroy(state);
  cmfo_state_destroy(next);
  cmfo_destroy(ctx);
  PASS();

  /* Test 8: Parse */
  TEST("parse");
  ctx = cmfo_init(NULL);
  cmfo_vec7_t parsed;
  result = cmfo_parse(ctx, "verdad", &parsed);
  assert(result == CMFO_OK);
  assert(fabs(parsed.v[1] - 1.0) < 1e-10); /* Truth axis */

  cmfo_destroy(ctx);
  PASS();

  printf("\n==========================\n");
  printf("All tests PASSED!\n");

  return 0;
}
