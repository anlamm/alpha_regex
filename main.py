'''

The following algorithm is from the paper :
    Synthesizing Regular Expressions from Examples for Introductory Automata Assignments - Mina Lee, Sunbeom So, Hakjoo Oh (2016)

Assumptions:
1. gree expression is regex expression

Algorithm highlight:
1. Uses placeholder to iteratively generate possible next state and do checking. Note that this is an infinite set, and it grows exponentially in terms of the depth of the search tree
   so pruning is required.
2. Over_approx is used to get the largest set that a state with placeholder can represent. If a valid_string is not in this set, then we can eliminate this state and its offsprings.
3. Similarly for under_approx, if an invalid_string is in this set, then we can eliminate this state and its offsprings.
4. There is also unroll_and_split algorithm introduced in the paper mentioned for further elimination (details are omitted here ...)

'''

import re
import heapq
from typing import List, Set, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import lru_cache

CACHE_SIZE = 10000
SMALL_CACHE_SIZE = 1024

def check_inputs(valid_strings: list[str], invalid_strings: list[str]) -> None:
    """
    check if the input is valid.

    Args:
        valid_strings (list[str]): list of strings whose input is valid.
        invalid_strings (list[str]): list of strings whose input is invalid.

    Returns:
        None

    Raises:
        ValueError: if the input is invalid.
    """
    # Check if valid_strings and invalid_strings are lists
    if not isinstance(valid_strings, list):
        raise TypeError("valid_strings must be a list")

    if not isinstance(invalid_strings, list):
        raise TypeError("invalid_strings must be a list")

    # Check if each list contains more than 5 elements
    if len(valid_strings) > 5:
        raise ValueError("valid_strings should not contain more than 5 elements")

    if len(invalid_strings) > 5:
        raise ValueError("invalid strings should not contain more than 5 elements")

    # Check if every element in both lists is string and has length <= 20
    for i in range(len(valid_strings)):
        if not isinstance(valid_strings[i], str):
            raise TypeError(
                f"All elements in valid_strings must be string.\n element with index {i} in valid_strings is not a string")
        if len(valid_strings[i]) > 20:
            raise ValueError(f"element with index {i} in valid_strings should not be longer than 20 characters")

    for i in range(len(invalid_strings)):
        if not isinstance(invalid_strings[i], str):
            raise TypeError(
                f"All elements in invalid_strings must be string.\n element with index {i} in invalid_strings is not a string")
        if len(invalid_strings[i]) > 20:
            raise ValueError(f"element with index {i} in invalid_strings should not be longer than 20 characters")


class State(ABC):
    """
    Base class for representing states.

    Attributes:
        cost (int): cost of the state alone
        total_cost (int): total cost of the state (including the substate it contains). Low total cost represents a simpler expression.
                          This is for retrieving simple state in the heap in workset.

    """
    cost: int
    total_cost: int

    def __lt__(self, other):
        if self.get_total_cost() == other.get_total_cost():
            if self is other:
                return self.lt(other)
            else:
                return True
        else:
            return self.get_total_cost() < other.get_total_cost()

    @abstractmethod
    def get_total_cost(self) -> int:
        pass

    @abstractmethod
    def lt(self, other):
        pass

@dataclass
class Placeholder(State):
    """A state representing a placeholder."""
    id: int
    cost = 6
    total_cost = 6

    def get_total_cost(self) -> int:
        return self.total_cost

    def lt(self, other):
        return self.id < other.id

    def __eq__(self, other):
        if not isinstance(other, Placeholder):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

@dataclass
class Alphanum(State):
    """A state representing a literal alphanumeric character or string."""
    string: str
    total_cost = 1
    cost = 1

    def get_total_cost(self) -> int:
        return self.total_cost

    def lt(self, other):
        return self.string < other.string

    def __eq__(self, other):
        if not isinstance(other, Alphanum):
            return False
        return self.string == other.string

    def __hash__(self):
        return hash(self.string)

@dataclass
class Or(State):
    """A state representing the OR operation (e.g., 'a|b')."""
    left: State
    right: State
    total_cost = -1
    cost = 3

    def get_total_cost(self) -> int:
        if self.total_cost == -1:
            self.total_cost = self.left.get_total_cost() + self.right.get_total_cost() + self.cost
        return self.total_cost

    def lt(self, other):
        return self.left < other.left

    def __eq__(self, other):
        if not isinstance(other, Or):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash(self.left) + hash(self.right)

@dataclass
class Concat(State):
    """A state representing concatenation (e.g., 'ab')."""
    left: State
    right: State
    total_cost = -1
    cost = 1

    def get_total_cost(self) -> int:
        if self.total_cost == -1:
            self.total_cost = self.left.get_total_cost() + self.right.get_total_cost() + self.cost
        return self.total_cost

    def lt(self, other):
        return self.left < other.left

    def __eq__(self, other):
        if not isinstance(other, Concat):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self):
        return hash(self.left) + hash(self.right)

@dataclass
class Star(State):
    """A state representing star (e.g., 'a*')."""
    value: State
    total_cost = -1
    cost = 7

    def get_total_cost(self) -> int:
        if self.total_cost == -1:
            self.total_cost = self.value.get_total_cost() + self.cost
        return self.total_cost

    def lt(self, other):
        return self.value < other.value

    def __eq__(self, other):
        if not isinstance(other, Star):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

@dataclass
class Exist(State):
    """A state representing an optional component (e.g., 'a?')."""
    value: State
    total_cost = -1
    cost = 6

    def get_total_cost(self) -> int:
        if self.total_cost == -1:
            self.total_cost = self.value.get_total_cost() + self.cost
        return self.total_cost

    def lt(self, other):
        return self.value < other.value

    def __eq__(self, other):
        if not isinstance(other, Exist):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)

@dataclass
class Plus(State):
    """A state representing one-or-more repetition (e.g., 'a+')."""
    value: State
    total_cost = -1
    cost = 7

    def get_total_cost(self) -> int:
        if self.total_cost == -1:
            self.total_cost = self.value.get_total_cost() + self.cost
        return self.total_cost

    def lt(self, other):
        return self.value < other.value

    def __eq__(self, other):
        if not isinstance(other, Plus):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


Placeholder_id = 0

def get_new_id() -> int:
    """
    Generate a unique integer ID for placeholders.
    """
    global Placeholder_id
    Placeholder_id += 1
    return Placeholder_id

def state2str(state: State) -> str:
    """
    Convert a state to a string representation of a gree expression.

    Args:
        state (State): The regex state to convert.

    Returns:
        str: The regex string representation, anchored with '^' and '$'.

    Raises:
        ValueError: If an unrecognized state type is encountered.
    """
    def state2str_recursion(inner_state: State) -> Optional[str]:
        """
        Recursively convert a state to a regex string without anchors.
        """
        if isinstance(inner_state, Placeholder):
            # Placeholder: Represented as a placeholder string (for debugging)
            return "PLACE_HOLDER"
        elif isinstance(inner_state, Alphanum):
            # Alphanum: Return the literal value (corrected from 'string' to 'value')
            return inner_state.string
        elif isinstance(inner_state, Or):
            # Or: Combine left and right operands with '|' in parentheses
            return f"({state2str_recursion(inner_state.left)}|{state2str_recursion(inner_state.right)})"
        elif isinstance(inner_state, Concat):
            # Concat: Concatenate left and right parts
            return f"{state2str_recursion(inner_state.left)}{state2str_recursion(inner_state.right)}"
        elif isinstance(inner_state, Star):
            # Star: Wrap value in parentheses with '*' suffix
            return f"({state2str_recursion(inner_state.value)})*"
        elif isinstance(inner_state, Exist):
            # Exist: Wrap value in parentheses with '?' suffix
            return f"({state2str_recursion(inner_state.value)})?"
        elif isinstance(inner_state, Plus):
            # Plus: Wrap value in parentheses with '+' suffix
            return f"({state2str_recursion(inner_state.value)})+"
        else:
            raise ValueError(f"unknown state type: {inner_state}")

    # Wrap the expression with '^' and '$'
    return "^" + state2str_recursion(state) + "$"

def is_solution(state: State, valid_strings: list[str], invalid_strings: list[str]) -> bool:
    exp = state2str(state)
    pos_valid = all(re.match(exp, string) for string in valid_strings)
    neg_valid = all((re.match(exp, string) is None for string in invalid_strings))
    return pos_valid and neg_valid

@lru_cache(maxsize=SMALL_CACHE_SIZE)
def find_placeholders(state: State) -> List[Placeholder]:
    if isinstance(state, Placeholder):
        return [state]
    elif isinstance(state, Or) or isinstance(state, Concat):
        left_result = find_placeholders(state.left)
        right_result = find_placeholders(state.right)
        return left_result + right_result
    elif isinstance(state, Star) or isinstance(state, Exist) or isinstance(state, Plus):
        return find_placeholders(state.value)
    else:
        return []

@lru_cache(maxsize=SMALL_CACHE_SIZE)
def subst_placeholder(state: State, avail: State, placeholder: State) -> State:
    if isinstance(placeholder, Placeholder):
        if isinstance(state, Placeholder) and state.id == placeholder.id:
            return avail
        elif isinstance(state, Placeholder) and state.id != placeholder.id:
            return state
        elif isinstance(state, Alphanum):
            return state
        elif isinstance(state, Or):
            left_result = subst_placeholder(state.left, avail, placeholder)
            right_result = subst_placeholder(state.right, avail, placeholder)
            return Or(left_result, right_result)
        elif isinstance(state, Concat):
            left_result = subst_placeholder(state.left, avail, placeholder)
            right_result = subst_placeholder(state.right, avail, placeholder)
            return Concat(left_result, right_result)
        elif isinstance(state, Star):
            inner_result = subst_placeholder(state.value, avail, placeholder)
            return Star(inner_result)
        elif isinstance(state, Exist):
            inner_result = subst_placeholder(state.value, avail, placeholder)
            return Exist(inner_result)
        elif isinstance(state, Plus):
            inner_result = subst_placeholder(state.value, avail, placeholder)
            return Plus(inner_result)
        else:
            raise ValueError(f"unknown state type: {state}")
    else:
        raise ValueError("No placeholder to substitute")

@lru_cache(maxsize=CACHE_SIZE)
def over_approx(state: State) -> State:
    """
    Generate an under-approximation of a regex state by replacing placeholders with an empty string.

    Args:
        state (State): The regex state to under-approximate.

    Returns:
        State: The under-approximated state.

    Raises:
        ValueError: If the state is unrecognized.
    """
    if isinstance(state, Placeholder):
        # Placeholder: Replace with '.*' to match any string
        return Alphanum(".*")
    elif isinstance(state, Alphanum):
        # Alphanum: Return unchanged as it is already a literal
        return state
    elif isinstance(state, Or):
        # Or: Recursively over-approximate both operands
        return Or(over_approx(state.left), over_approx(state.right))
    elif isinstance(state, Concat):
        # Concat: Recursively over-approximate both parts
        return Concat(over_approx(state.left), over_approx(state.right))
    elif isinstance(state, Star):
        # Star: Recursively over-approximate the repeated value
        return Star(over_approx(state.value))
    elif isinstance(state, Exist):
        # Exist: Recursively over-approximate the optional value
        return Exist(over_approx(state.value))
    elif isinstance(state, Plus):
        # Plus: Recursively over-approximate the repeated value
        return Plus(over_approx(state.value))
    else:
        raise ValueError(f"unknown state type: {state}")

@lru_cache(maxsize=CACHE_SIZE)
def under_approx(state: State) -> State:
    """
    Generate an under-approximation of a regex state by replacing placeholders with an empty string.

    Args:
        state (State): The regex state to under-approximate.

    Returns:
        State: The under-approximated state.

    Raises:
        ValueError: If the state is unrecognized.
    """
    if isinstance(state, Placeholder):
        # Placeholder: Replace with empty string to match nothing
        return Alphanum("")
    elif isinstance(state, Alphanum):
        # Alphanum: Return unchanged as it is already a literal
        return state
    elif isinstance(state, Or):
        # Or: Recursively under-approximate both operands
        return Or(under_approx(state.left), under_approx(state.right))
    elif isinstance(state, Concat):
        # Concat: Recursively under-approximate both parts
        return Concat(under_approx(state.left), under_approx(state.right))
    elif isinstance(state, Star):
        # Star: Recursively under-approximate the repeated value
        return Star(under_approx(state.value))
    elif isinstance(state, Exist):
        # Exist: Recursively under-approximate the optional value
        return Exist(under_approx(state.value))
    elif isinstance(state, Plus):
        # Plus: Recursively under-approximate the repeated value
        return Plus(under_approx(state.value))
    else:
        raise ValueError(f"unknown state type: {state}")

@lru_cache(maxsize=CACHE_SIZE)
def unroll(state: State) -> State:
    """
    Transform a state by unrolling complex constructs into simpler forms.

    Args:
        state (State): The state to unroll.

    Returns:
        State: The unrolled state.

    Raises:
        ValueError: If the state is uncognized.
    """
    if isinstance(state, Placeholder):
        # unroll(placeholder) = placeholder
        return state
    elif isinstance(state, Alphanum):
        # unroll(Alphanum) = Alphanum
        return state
    elif isinstance(state, Or):
        # unroll(Or(a, b)) = Or(unroll(a), unroll(b))
        return Or(unroll(state.left), unroll(state.right))
    elif isinstance(state, Concat):
        # unroll(Concat(a, b)) = Concat(unroll(a), unroll(b))
        return Concat(unroll(state.left), unroll(state.right))
    elif isinstance(state, Exist):
        # Exist(a) = Or("", a)
        return unroll(Or(Alphanum(""), state.value))
    elif isinstance(state, Plus):
        # Plus(a) = Concat(a, a*)
        return unroll(Concat(state.value, Star(state.value)))
    elif isinstance(state, Star):
        # unroll(a*) = Concat(Concat(a, a), a*)
        return Concat(Concat(state.value, state.value), state)
    else:
        raise ValueError(f"unknown state type: {state}")

@lru_cache(maxsize=CACHE_SIZE)
def split(state: State) -> Set[State]:
    """
    Decompose a state into a set of possible substates for exploration.

    Args:
        state (State): The state to split.

    Returns:
        Set[State]: The set of possible substates.

    Raises:
        ValueError: If the state is uncognized.
    """
    if isinstance(state, Placeholder):
        # split(placeholder) = {placeholder}
        return {Placeholder(get_new_id())}
    elif isinstance(state, Alphanum) or isinstance(state, Star):
        # split(Alphanum) = {Alphanum} ; split(a*) = {a*}
        return {state}
    elif isinstance(state, Or):
        # split(Or(a, b)) = Union(split(a), split(b))
        return split(state.left) | split(state.right)
    elif isinstance(state, Concat):
        # split(Concat(a, b)) = {Concat(a', b), Concat(a, b') for a' \in split(a) and b' in split(b)}
        split_set = set()
        for s in split(state.left):
            split_set.add(Concat(s, state.right))
        for s in split(state.right):
            split_set.add(Concat(state.left, s))
        return split_set
    elif isinstance(state, Exist):
        # Exist(a) = Or("", a)
        return split(Or(Alphanum(""), state.value))
    elif isinstance(state, Plus):
        # Plus(a) = Concat(a, a*)
        return split(Concat(state.value, Star(state.value)))
    else:
        raise ValueError(f"unknown state type: {state}")


class Workset:
    """
    Allow store and retrieve explored and possible solution states

    Attributes:
        heap : a min heap that stores tuple of the form (cost, (state, placeholder))
        explored set: a set that stored the explored states
    """
    def __init__(self):
        self.heap: List[Tuple[int, Tuple[State, Optional[State]]]] = []  # (cost, (state, placeholder))
        self.explored_set: Set[str] = set()

    def add(self, state_tuple: Tuple[State, Optional[State]]) -> None:
        """
        Add a state tuple (state, placeholder) to the heap in the workset if not already explored.
        """
        state, placeholder = state_tuple
        state_str = state2str(state)
        if state_str not in self.explored_set:
            cost = state.get_total_cost()
            heapq.heappush(self.heap, (cost, state_tuple))
            self.explored_set.add(state_str)

    def choose(self) -> Optional[Tuple[State, Optional[State]]]:
        """
        Pop the min cost state tuple from the heap
        """
        if not self.heap:
            return None
        cost, state_tuple = heapq.heappop(self.heap)
        return state_tuple

    def next(self, state: State, placeholder: State, valid_strings: list[str], invalid_strings: list[str], availables: list[State]) -> None:
        """
        Validate new state by replacing placeholder with possible components from available. Then, add to the workset.
        """
        is_invalid = False

        over_approx_new_state = over_approx(state)
        for string in valid_strings:
            if not re.match(state2str(over_approx_new_state), string):
                is_invalid = True
                break
        if is_invalid:
            return
        under_approx_new_state = under_approx(state)
        for string in invalid_strings:
            if re.match(state2str(under_approx_new_state), string):
                is_invalid = True
                break
        if is_invalid:
            return

        states_to_be_added = set()
        for avail in availables:
            new_state = subst_placeholder(state, avail, placeholder)
            split_unroll_states: Set[State] = split(unroll(state))
            for su_state in split_unroll_states:
                is_invalid = True
                over_su_state = over_approx(su_state)
                for string in valid_strings:
                    if re.match(state2str(over_su_state), string):
                        is_invalid = False
                        break
                if is_invalid:
                    return

            new_placeholders = find_placeholders(new_state)
            new_state_tuple: Tuple[State, Optional[State]] = (new_state, new_placeholders[0] if new_placeholders else None)
            states_to_be_added.add(new_state_tuple)

        for add_state in states_to_be_added:
            self.add(add_state)

def get_characters(valid_strings: list[str]) -> list[State]:
    """
    Extract possible state components from valid_strings.
    """
    char_from_strings = []
    for string in valid_strings:
        for c in string:
            if c not in char_from_strings:
                char_from_strings.append(Alphanum(c))

    default = [
        Alphanum("[A-Z]"), Alphanum("[a-z]"), Alphanum("[0-9]"), Alphanum("\w"), Alphanum("\D"), Alphanum("\\n"), Alphanum("."),
        Or(Placeholder(get_new_id()), Placeholder(get_new_id())), Concat(Placeholder(get_new_id()), Placeholder(get_new_id())),
        Star(Placeholder(get_new_id())), Exist(Placeholder(get_new_id())), Plus(Placeholder(get_new_id()))
    ]

    char_from_strings.extend(default)
    return char_from_strings

def generate_gree_expression(valid_strings: list[str], invalid_strings: list[str]) -> str:
    """
    Generate a gree expression that matches valid strings and excludes invalid ones.

    This function implements a search-based approach to construct a "gree expression" (a regex-like
    pattern) that matches all strings in `valid_strings` while ensuring none of the strings in
    `invalid_strings` are matched. It uses a workset to explore possible state combinations,
    starting with a placeholder state, and iteratively refines it using available components.

    Args:
        valid_strings (List[str]): A list of strings that the generated expression must match.
        invalid_strings (List[str]): A list of strings that the generated expression must not match.

    Returns:
        str: A string representing the generated regex-like expression, or "no solution" if no valid
             expression can be found.
    """
    # Check input
    check_inputs(valid_strings, invalid_strings)

    # Get possible components to form the gree expression
    availables: List[State] = get_characters(valid_strings)

    w = Workset()
    init_state = Placeholder(get_new_id())
    w.add((init_state, init_state))   # start with a placeholder

    while True:
        # Get a possible solution state with the smallest total cost
        result = w.choose()

        if not result:
            # if no possible solution state, then stop searching
            break

        state, placeholder = result
        if placeholder:
            # if there is a placeholder in the state, replace it with possible components from availables and add to the workset
            w.next(state, placeholder, valid_strings, invalid_strings, availables)
        elif is_solution(state, valid_strings, invalid_strings):
            # if the state is solution, convert it to string and return
            return state2str(state)

    return "no solution"