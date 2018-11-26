When play begins, seed the random-number generator with 1234.

supporter is a kind of thing.
container is a kind of thing.
object-like is a kind of thing.
door is a kind of thing.
key is a kind of object-like.
food is a kind of object-like.
supporters are fixed in place.
containers are openable, lockable and fixed in place. containers are usually closed.
object-like is portable.
door is openable and lockable.
food is edible.
A room has a text called internal name.


The r_3 and the r_2 and the r_0 and the r_1 and the r_4 are rooms.

The internal name of r_3 is "attic".
The printed name of r_3 is "-= Attic =-".
The attic part 0 is some text that varies. The attic part 0 is "Well how about that, you are in a place we're calling an attic.

 [if c_0 is locked]A locked[else if c_0 is open]An open[otherwise]A closed[end if]".
The attic part 1 is some text that varies. The attic part 1 is " ordinary looking locker is in the room.[if c_0 is open and there is something in the c_0] The locker contains [a list of things in the c_0].[end if]".
The attic part 2 is some text that varies. The attic part 2 is "[if c_0 is open and the c_0 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The attic part 3 is some text that varies. The attic part 3 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The attic part 4 is some text that varies. The attic part 4 is " gate leading west. There is an unguarded exit to the south.".
The description of r_3 is "[attic part 0][attic part 1][attic part 2][attic part 3][attic part 4]".

west of r_3 and east of r_2 is a door called d_0.
The r_4 is mapped south of r_3.
The internal name of r_2 is "scullery".
The printed name of r_2 is "-= Scullery =-".
The scullery part 0 is some text that varies. The scullery part 0 is "You've just shown up in a scullery.

 You make out a formless locker.[if c_2 is open and there is something in the c_2] The formless locker contains [a list of things in the c_2].[end if]".
The scullery part 1 is some text that varies. The scullery part 1 is "[if c_2 is open and the c_2 contains nothing] The formless locker is empty! What a waste of a day![end if]".
The scullery part 2 is some text that varies. The scullery part 2 is " You can make out [if c_3 is locked]a locked[else if c_3 is open]an opened[otherwise]a closed[end if]".
The scullery part 3 is some text that varies. The scullery part 3 is " case.[if c_3 is open and there is something in the c_3] The case contains [a list of things in the c_3].[end if]".
The scullery part 4 is some text that varies. The scullery part 4 is "[if c_3 is open and the c_3 contains nothing] The case is empty! This is the worst thing that could possibly happen, ever![end if]".
The scullery part 5 is some text that varies. The scullery part 5 is " You bend down to tie your shoe. When you stand up, you notice a table. [if there is something on the s_0]You see [a list of things on the s_0] on the table.[end if]".
The scullery part 6 is some text that varies. The scullery part 6 is "[if there is nothing on the s_0]Looks like someone's already been here and taken everything off it, though.[end if]".
The scullery part 7 is some text that varies. The scullery part 7 is "

 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The scullery part 8 is some text that varies. The scullery part 8 is " door leading south. There is [if d_0 is open]an open[otherwise]a closed[end if]".
The scullery part 9 is some text that varies. The scullery part 9 is " gate leading east.".
The description of r_2 is "[scullery part 0][scullery part 1][scullery part 2][scullery part 3][scullery part 4][scullery part 5][scullery part 6][scullery part 7][scullery part 8][scullery part 9]".

south of r_2 and north of r_1 is a door called d_1.
east of r_2 and west of r_3 is a door called d_0.
The internal name of r_0 is "pantry".
The printed name of r_0 is "-= Pantry =-".
The pantry part 0 is some text that varies. The pantry part 0 is "You've just shown up in a pantry. You begin to take stock of what's here.



There is an exit to the north. Don't worry, it is unblocked.".
The description of r_0 is "[pantry part 0]".

The r_1 is mapped north of r_0.
The internal name of r_1 is "bedchamber".
The printed name of r_1 is "-= Bedchamber =-".
The bedchamber part 0 is some text that varies. The bedchamber part 0 is "This is going to sound unbelievable, but you've just entered a bedchamber. The room seems oddly familiar, as though it were only superficially different from the other rooms in the building.

 [if c_4 is locked]A locked[else if c_4 is open]An open[otherwise]A closed[end if]".
The bedchamber part 1 is some text that varies. The bedchamber part 1 is " chest is in the corner.[if c_4 is open and there is something in the c_4] The chest contains [a list of things in the c_4]. You look at the price tag that hangs off the chest. Sixty dollars?! Where'd they buy this chest, some kind of expensive place?[end if]".
The bedchamber part 2 is some text that varies. The bedchamber part 2 is "[if c_4 is open and the c_4 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The bedchamber part 3 is some text that varies. The bedchamber part 3 is "

 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The bedchamber part 4 is some text that varies. The bedchamber part 4 is " door leading north. There is an unguarded exit to the south.".
The description of r_1 is "[bedchamber part 0][bedchamber part 1][bedchamber part 2][bedchamber part 3][bedchamber part 4]".

The r_0 is mapped south of r_1.
north of r_1 and south of r_2 is a door called d_1.
The internal name of r_4 is "vault".
The printed name of r_4 is "-= Vault =-".
The vault part 0 is some text that varies. The vault part 0 is "Well, here we are in the vault. You decide to just list off a complete list of everything you see in the room, because hey, why not?

 You can see [if c_1 is locked]a locked[else if c_1 is open]an open[otherwise]a closed[end if]".
The vault part 1 is some text that varies. The vault part 1 is " toolbox nearby.[if c_1 is open and there is something in the c_1] The toolbox contains [a list of things in the c_1].[end if]".
The vault part 2 is some text that varies. The vault part 2 is "[if c_1 is open and the c_1 contains nothing] What a letdown! The toolbox is empty![end if]".
The vault part 3 is some text that varies. The vault part 3 is " You make out [if c_5 is locked]a locked[else if c_5 is open]an opened[otherwise]a closed[end if]".
The vault part 4 is some text that varies. The vault part 4 is " crate.[if c_5 is open and there is something in the c_5] The crate contains [a list of things in the c_5]. I mean, just wow! Isn't TextWorld just the best?[end if]".
The vault part 5 is some text that varies. The vault part 5 is "[if c_5 is open and the c_5 contains nothing] The crate is empty! This is the worst thing that could possibly happen, ever![end if]".
The vault part 6 is some text that varies. The vault part 6 is "

There is an exit to the north. Don't worry, it is unguarded.".
The description of r_4 is "[vault part 0][vault part 1][vault part 2][vault part 3][vault part 4][vault part 5][vault part 6]".

The r_3 is mapped north of r_4.

The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 are containers.
The c_0 and the c_1 and the c_2 and the c_3 and the c_4 and the c_5 are privately-named.
The d_1 and the d_0 are doors.
The d_1 and the d_0 are privately-named.
The k_2 and the k_0 and the k_1 and the k_5 are keys.
The k_2 and the k_0 and the k_1 and the k_5 are privately-named.
The o_0 and the o_1 are object-likes.
The o_0 and the o_1 are privately-named.
The r_3 and the r_2 and the r_0 and the r_1 and the r_4 are rooms.
The r_3 and the r_2 and the r_0 and the r_1 and the r_4 are privately-named.
The s_0 are supporters.
The s_0 are privately-named.

The description of d_1 is "The door looks commanding. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_1 is "door".
Understand "door" as d_1.
The d_1 is closed.
The description of d_0 is "it's a robust gate [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_0 is "gate".
Understand "gate" as d_0.
The d_0 is open.
The description of c_0 is "The locker looks strong, and impossible to force open. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_0 is "locker".
Understand "locker" as c_0.
The c_0 is in r_3.
The c_0 is open.
The description of c_1 is "The toolbox looks strong, and impossible to open. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_1 is "toolbox".
Understand "toolbox" as c_1.
The c_1 is in r_4.
The c_1 is open.
The description of c_2 is "The formless locker looks strong, and impossible to crack open. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_2 is "formless locker".
Understand "formless locker" as c_2.
Understand "formless" as c_2.
Understand "locker" as c_2.
The c_2 is in r_2.
The c_2 is closed.
The description of c_3 is "The case looks strong, and impossible to force open. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_3 is "case".
Understand "case" as c_3.
The c_3 is in r_2.
The c_3 is locked.
The description of c_4 is "The chest looks strong, and impossible to force open. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_4 is "chest".
Understand "chest" as c_4.
The c_4 is in r_1.
The c_4 is locked.
The description of c_5 is "The crate looks strong, and impossible to open. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_5 is "crate".
Understand "crate" as c_5.
The c_5 is in r_4.
The c_5 is locked.
The description of k_2 is "The formless keycard looks useful".
The printed name of k_2 is "formless keycard".
Understand "formless keycard" as k_2.
Understand "formless" as k_2.
Understand "keycard" as k_2.
The k_2 is in r_2.
The matching key of the c_2 is the k_2.
The description of s_0 is "The table is an unstable piece of trash.".
The printed name of s_0 is "table".
Understand "table" as s_0.
The s_0 is in r_2.
The description of k_0 is "The metal of the key is hammered.".
The printed name of k_0 is "key".
Understand "key" as k_0.
The k_0 is in the c_1.
The description of k_1 is "The metal of the keycard is hammered.".
The printed name of k_1 is "keycard".
Understand "keycard" as k_1.
The player carries the k_1.
The matching key of the c_0 is the k_1.
The description of k_5 is "The passkey looks useful".
The printed name of k_5 is "passkey".
Understand "passkey" as k_5.
The player carries the k_5.
The matching key of the c_4 is the k_5.
The description of o_0 is "The cloak seems to fit in here".
The printed name of o_0 is "cloak".
Understand "cloak" as o_0.
The player carries the o_0.
The description of o_1 is "The lampshade is well-used.".
The printed name of o_1 is "lampshade".
Understand "lampshade" as o_1.
The player carries the o_1.


The player is in r_2.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "go east / go south / take key from toolbox / go north / insert key into locker"

Every turn:
	if quest0 completed is true:
		say "";
	else if The player is in r_3 and The c_0 is in r_3 and The c_0 is open and The k_0 is in the c_0:
		increase the score by 1; [Quest completed]
		Now the quest0 completed is true;

Use scoring. The maximum score is 1.
This is the simpler notify score changes rule:
	If the score is not the last notified score:
		let V be the score - the last notified score;
		say "Your score has just gone up by [V in words] ";
		if V > 1:
			say "points.";
		else:
			say "point.";
		Now the last notified score is the score;
	if score is maximum score:
		end the story finally; [Win]

The simpler notify score changes rule substitutes for the notify score changes rule.

Rule for listing nondescript items:
	stop.

Rule for printing the banner text:
	say "You are now playing a exciting round of TextWorld! Here is how to play! First of all, go east. With that done, make an attempt to head south. And then, pick up the key from the toolbox. Then, attempt to go to the north. After that, you can deposit the key inside the locker inside the attic. Alright, thanks![line break]".

Include Basic Screen Effects by Emily Short.

Rule for printing the player's obituary:
	if story has ended finally:
		center "*** The End ***";
	else:
		center "*** You lost! ***";
	say paragraph break;
	let X be the turn count;
	if restrict commands option is true:
		let X be the turn count minus one;
	say "You scored [score] out of a possible [maximum score], in [X] turn(s).";
	[wait for any key;
	stop game abruptly;]
	rule succeeds.

Rule for implicitly taking something (called target):
	if target is fixed in place:
		say "The [target] is fixed in place.";
	otherwise:
		say "You need to take the [target] first.";
	stop.

Does the player mean doing something:
	if the noun is not nothing and the second noun is nothing and the player's command matches the text printed name of the noun:
		it is likely;
	if the noun is nothing and the second noun is not nothing and the player's command matches the text printed name of the second noun:
		it is likely;
	if the noun is not nothing and the second noun is not nothing and the player's command matches the text printed name of the noun and the player's command matches the text printed name of the second noun:
		it is very likely.  [Handle action with two arguments.]
Printing the content of the room is an activity.
Rule for printing the content of the room:
	let R be the location of the player;
	say "Room contents:[line break]";
	list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the world is an activity.
Rule for printing the content of the world:
	let L be the list of the rooms;
	say "World: [line break]";
	repeat with R running through L:
		say "  [the internal name of R][line break]";
	repeat with R running through L:
		say "[the internal name of R]:[line break]";
		if the list of things in R is empty:
			say "  nothing[line break]";
		otherwise:
			list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the inventory is an activity.
Rule for printing the content of the inventory:
	say "Inventory:[line break]";
	list the contents of the player, with newlines, indented, giving inventory information, including all contents, with extra indentation.

Printing the content of nowhere is an activity.
Rule for printing the content of nowhere:
	say "Nowhere:[line break]";
	let L be the list of the off-stage things;
	repeat with thing running through L:
		say "  [thing][line break]";

Printing the things on the floor is an activity.
Rule for printing the things on the floor:
	let R be the location of the player;
	let L be the list of things in R;
	remove yourself from L;
	remove the list of containers from L;
	remove the list of supporters from L;
	remove the list of doors from L;
	if the number of entries in L is greater than 0:
		say "There is [L with indefinite articles] on the floor.";

After printing the name of something (called target) while
printing the content of the room
or printing the content of the world
or printing the content of the inventory
or printing the content of nowhere:
	follow the property-aggregation rules for the target.

The property-aggregation rules are an object-based rulebook.
The property-aggregation rulebook has a list of text called the tagline.

[At the moment, we only support "open/unlocked", "closed/unlocked" and "closed/locked" for doors and containers.]
[A first property-aggregation rule for an openable open thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable closed thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an lockable unlocked thing (this is the mention unlocked lockable rule):
	add "unlocked" to the tagline.

A property-aggregation rule for an lockable locked thing (this is the mention locked lockable rule):
	add "locked" to the tagline.]

A first property-aggregation rule for an openable lockable open unlocked thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable lockable closed unlocked thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an openable lockable closed locked thing (this is the mention locked openables rule):
	add "locked" to the tagline.

A property-aggregation rule for a lockable thing (called the lockable thing) (this is the mention matching key of lockable rule):
	let X be the matching key of the lockable thing;
	if X is not nothing:
		add "match [X]" to the tagline.

A property-aggregation rule for an edible off-stage thing (this is the mention eaten edible rule):
	add "eaten" to the tagline.

The last property-aggregation rule (this is the print aggregated properties rule):
	if the number of entries in the tagline is greater than 0:
		say " ([tagline])";
		rule succeeds;
	rule fails;

An objective is some text that varies. The objective is "You are now playing a exciting round of TextWorld! Here is how to play! First of all, go east. With that done, make an attempt to head south. And then, pick up the key from the toolbox. Then, attempt to go to the north. After that, you can deposit the key inside the locker inside the attic. Alright, thanks!".
Printing the objective is an action applying to nothing.
Carry out printing the objective:
	say "[objective]".

Understand "goal" as printing the objective.

The taking action has an object called previous locale (matched as "from").

Setting action variables for taking:
	now previous locale is the holder of the noun.

Report taking something from the location:
	say "You pick up [the noun] from the ground." instead.

Report taking something:
	say "You take [the noun] from [the previous locale]." instead.

Report dropping something:
	say "You drop [the noun] on the ground." instead.

The print state option is a truth state that varies.
The print state option is usually false.

Turning on the print state option is an action applying to nothing.
Carry out turning on the print state option:
	Now the print state option is true.

Turning off the print state option is an action applying to nothing.
Carry out turning off the print state option:
	Now the print state option is false.

Printing the state is an activity.
Rule for printing the state:
	let R be the location of the player;
	say "Room: [line break] [the internal name of R][line break]";
	[say "[line break]";
	carry out the printing the content of the room activity;]
	say "[line break]";
	carry out the printing the content of the world activity;
	say "[line break]";
	carry out the printing the content of the inventory activity;
	say "[line break]";
	carry out the printing the content of nowhere activity;
	say "[line break]".

Printing the entire state is an action applying to nothing.
Carry out printing the entire state:
	say "-=STATE START=-[line break]";
	carry out the printing the state activity;
	say "[line break]Score:[line break] [score]/[maximum score][line break]";
	say "[line break]Objective:[line break] [objective][line break]";
	say "[line break]Inventory description:[line break]";
	say "  You are carrying: [a list of things carried by the player].[line break]";
	say "[line break]Room description:[line break]";
	try looking;
	say "[line break]-=STATE STOP=-";

When play begins:
	if print state option is true:
		try printing the entire state;

Every turn:
	if print state option is true:
		try printing the entire state;

When play ends:
	if print state option is true:
		try printing the entire state;

After looking:
	carry out the printing the things on the floor activity.

Understand "print_state" as printing the entire state.
Understand "enable print state option" as turning on the print state option.
Understand "disable print state option" as turning off the print state option.

Before going through a closed door (called the blocking door):
	say "You have to open the [blocking door] first.";
	stop.

Before opening a locked door (called the locked door):
	let X be the matching key of the locked door;
	if X is nothing:
		say "The [locked door] is welded shut.";
	otherwise:
		say "You have to unlock the [locked door] with the [X] first.";
	stop.

Before opening a locked container (called the locked container):
	let X be the matching key of the locked container;
	if X is nothing:
		say "The [locked container] is welded shut.";
	otherwise:
		say "You have to unlock the [locked container] with the [X] first.";
	stop.

Displaying help message is an action applying to nothing.
Carry out displaying help message:
	say "[fixed letter spacing]Available commands:[line break]";
	say "  look:                                describe the current room[line break]";
	say "  goal:                                print the goal of this game[line break]";
	say "  inventory:                           print player's inventory[line break]";
	say "  go <dir>:                            move the player north, east, south or west[line break]";
	say "  examine <something>:                 examine something more closely[line break]";
	say "  eat <something>:                     eat something edible[line break]";
	say "  open <something>:                    open a door or a container[line break]";
	say "  close <something>:                   close a door or a container[line break]";
	say "  drop <something>:                    drop an object on the floor[line break]";
	say "  take <something>:                    take an object that is on the floor[line break]";
	say "  put <something> on <something>:      place an object on a supporter[line break]";
	say "  take <something> from <something>:   take an object from a container or a supporter[line break]";
	say "  insert <something> into <something>: place an object into a container[line break]";
	say "  lock <something> with <something>:   lock a door or a container with a key[line break]";
	say "  unlock <something> with <something>: unlock a door or a container with a key[line break]";

Understand "help" as displaying help message.

Taking all is an action applying to nothing.
Carry out taking all:
	say "You have to be more specific!".

Understand "take all" as taking all.
Understand "get all" as taking all.
Understand "pick up all" as taking all.

Understand "take each" as taking all.
Understand "get each" as taking all.
Understand "pick up each" as taking all.

Understand "take everything" as taking all.
Understand "get everything" as taking all.
Understand "pick up everything" as taking all.

The restrict commands option is a truth state that varies.
The restrict commands option is usually false.

Turning on the restrict commands option is an action applying to nothing.
Carry out turning on the restrict commands option:
	Decrease turn count by 1;
	Now the restrict commands option is true.

Understand "restrict commands" as turning on the restrict commands option.

The taking allowed flag is a truth state that varies.
The taking allowed flag is usually false.

Before removing something from something:
	now the taking allowed flag is true.

After removing something from something:
	now the taking allowed flag is false.

Before taking a thing (called the object) when the object is on a supporter (called the supporter):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [supporter] instead.";
		rule fails.

Before of taking a thing (called the object) when the object is in a container (called the container):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [container] instead.";
		rule fails.

Reporting max score is an action applying to nothing.
Carry out reporting max score:
	say "[maximum score]".

Understand "tw-print max_score" as reporting max score.

There is a EndOfObject.

