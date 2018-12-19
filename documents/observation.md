## Observation notebook

### 1 obs.reward

an integer, to receive reward.

**type**: `numpy.int32`

### 2. obs.observation.available_actions

a list of available actions

**type:** list

### 3.obs.observation.single_select

a list

**type:** （7）list

- unit type(interesting)
- player_relative(interesting)
- health(interesting)
- shields
- energy
- transport slot taken if it's in a transport
- build progress as a percentage if it's still being built

### 4.obs.observation.multi_select

**type:** （n, 7）list

### 5.obs.observation.feature_minimap

**type:** （7, 64, 64）list

**useful**(index-semantics):

1. 3-camera
2. 5-player_relative
3. 6-selected

### 6.obs.observation.feature_screen

**type:** (17, 84, 84)list

**useful**(index-semantics):

1. 1-visibility
2. 5-player_relative
3. 6-unit_type
4. 7-screen-selected
5. 8-hit_points
6. 9-hit_points_ratio
7. 14-density
8. 15-density-aa

### 7.obs.observation.player

A `(11)` tensor showing general information.

- player_id(interesting)
- minerals(interesting)
- vespene
- food used (otherwise known as supply)
- food cap
- food used by army
- food used by workers
- idle worker count
- army count(interesting)
- warp gate count (for protoss)
- larva count (for zerg)