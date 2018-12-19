## Action 

**Coordination:**

- Coordination system in pysc2 starts from left-top point
- First index is to the horizontal axis
- Second index is to the  vertical axis

### 1 No operator

<u>id</u>: 0

<u>semantic:</u> do not take any action

<u>implementation:</u>  `actions.FunctionCall(0, [])`

fc_layer + 1

### 2 move camera

<u>id</u>: 1

<u>semantic:</u> move camera

<u>implementation:</u>  `actions.FunctionCall(1, [[0~63], [0~63]])`

fc_layer + 1

conv_64 + 1

### 3 select rectangle

<u>id</u>: 3

<u>semantic:</u> select a rectangle between two points

<u>implementation:</u>  `actions.FunctionCall(3, [[0~1], [0~83, 0~83], [0~83, 0~83]])`

0~select, 1~add

fc_layer + 1(whether take?) + 2 (first_parameter)

conv_84 + 2

### 4 Train_Marine_quick

<u>id</u>: 477

<u>semantic:</u> train a marine soilder immediately

<u>implementation:</u>  `actions.FunctionCall(477, [[True]])`

fc_layer + 1

### 5 Attack_screen

<u>id</u>: 12

<u>semantic:</u> attack a point in screen

<u>implementation:</u>  `FUNCTIONS.Attack_screen("now", [0~83, 0~83])`

fc_layer + 1

conv_84 + 1

### 6 Attack_minimap

<u>id</u>: 13

<u>semantic:</u> attack a point in minimap

<u>implementation:</u>  `actions.FunctionCall(13, [[0/1], [0~63, 0~63]])`

0/1 whether queue the action

fc_layer  +  1(whether take?) + 2 (first_parameter)

conv_64 + 1

### 7 Move_screen

<u>id</u>: 331

<u>semantic:</u> move units selected to one place

<u>implementation:</u>  `actions.FunctionCall(331, [[0/1], [0~83, 0~83]])`

0/1 whether queue the action

fc_layer  +  1(whether take?) + 2 (first_parameter)

conv_84 + 1

### 8 Move_minimap

<u>id</u>: 332

<u>semantic:</u> move units selected to one place in minimap

<u>implementation:</u>  `actions.FunctionCall(332, [[0/1], [0~63, 0~63]])`

fc_layer  +  1(whether take?) + 2 (first_parameter)

conv_64 + 1

### 9 hold position

<u>id</u>: 274

<u>semantic</u>: hold position

<u>implementation:</u>  `actions.FunctionCall(274, [[0/1]])`

fc_layer  +  1(whether take?) + 2 (first_parameter)

### 10 Effect_Heal_screen 

<u>id</u>: 198

<u>semantic</u>: effect heal screen

<u>implementation:</u>  `actions.FunctionCall(198, [[0/1], [0~83, 0~83]])`

fc_layer  +  1(whether take?) + 2 (first_parameter)

conv_84 + 1

### 11 Effect_Heal_autocast 

<u>id</u>: 199

<u>semantic</u>: effect heal autocast

<u>implementation:</u>  `actions.FunctionCall(199, [])`

fc_layer  +  1(whether take?)

### 12 select_point

<u>id</u>: 2

<u>semantic</u>: select

<u>implementation:</u>  `actions.FunctionCall(2, [[0~3],[0~83, 0~83]])`

- 0-select
- 1-toggle
- 2-select_all_type
- 3-add_all_type

fc_layer  +  1(whether take?) + 4 (first_parameter)

conv_84 + 1