"""This is the base step class that all processing steps are derived from.

"""
import os
from copy import deepcopy
import pickle


class base_step(object):
    def __init__(self, persistent_directory=None):
        self.persistent_directory = persistent_directory
        if persistent_directory is not None:
            os.makedirs(self.persistent_directory, exist_ok=True)

        # we need to identify steps, e.g. for checking if required, previous
        # steps have already run. No space, no fancy characters
        self.name = None

        # A user-readable title for this step
        self.title = None

        # identifier for the help page to show when showing the associated gui
        self.help_page = None

        # we have a few status variables
        # this indicates if the step has run successfully at least once
        self.has_run = False

        # we differentiate between settings that have already been applied
        # (i.e., those that correspond to self.results), and those that will be
        # applied to generate new results at some time
        self.input_applied = {}
        self.input_new = {}

        # This variable will be populated with a dictionary of the same
        # structure as self.input_applied and self.input_new. Instead of actual
        # values, the key:value pairs hold the data types of the items. This
        # way we a) have a reference for the expected input structure and b)
        # could implement a validity check
        self.input_skel = None

        # we store results of this step here
        self.results = {}

        # steps are stored in a linked list (or tree? We allow multiple next
        # items)
        self.next_step = []
        self.prev_step = None

        # these steps must reside somewhere in the prev-branch and be finished
        # before this step can run
        # Set to None if no steps are required for this one
        self.required_steps = []

        # we inherently provide a Jupyter Widget-based GUI to each step
        # this gui element can be embedded into larger notebook guis, e.g. for
        # a complete workflow
        self.jupyter_gui = None
        self.widgets = {}

        # this callback can be used to trigger events outside of this step
        # object
        self.callback_step_ran = None

    def can_run(self):
        """Check if all required steps have been finished
        """
        if self.required_steps is None:
            return True

        def find_required_steps(branch, search_results):
            if branch.name in search_results.keys():
                search_results[branch.name] = branch.has_run

            if branch.prev_step is not None:
                search_results = find_required_steps(
                    branch.prev_step, search_results
                )
            return search_results

        search_results = find_required_steps(
            self.prev_step,
            {key: None for key in self.required_steps}
        )
        print('Search results:')
        print(search_results)
        can_run = True
        for key, item in search_results.items():
            if item is None:
                print('[{}] Required step not found: {}'.format(
                    self.name, key
                ))
                return False
            print('testing:', can_run, item)
            can_run = can_run & item
            print('   result:', can_run)
        return can_run

    def set_input_new(self, input_new):
        """Apply a new set of inputs

        TODO: This is the place to check the input_new dictionary for
        consistency with self.input_skel

        """
        assert isinstance(input_new, dict), "input_new must be a dict"
        self.input_new = input_new

    def transfer_input_new_to_applied(self):
        """Make a copy of the self.input_new dict and store in
        self.input_applied

        This is complicated because some objects cannot be easily copied (e.g.,
        io.BytesIO). Therefore, each step must implement this function by
        itself.
        """
        # this should suffice for simple input dicts
        self.input_applied = deepcopy(self.input_new)

    def apply_next_input(self):
        """Actually execute the step based in self.input_new

        Parameters
        ----------


        Returns
        -------
        ret_value: bool
            True if input was successfully applied, False if not

        """
        # ALWAYS set the variable to either True or False
        self.has_run = True
        raise Exception('Must be implemented by deriving class')

    def create_ipywidget_gui(self):
        """

        """
        raise Exception('Must be implemented by deriving class')

    def persistency_store(self):
        """Store the current state of the widget in the given persistency
        directory
        """
        print('Persistent store - base version', self.name)
        if self.persistent_directory is None:
            return

        stepdir = self.persistent_directory + os.sep + 'step_' + self.name
        print('Writing data to:', stepdir)
        os.makedirs(stepdir, exist_ok=True)

        # store settings in pickle
        with open(stepdir + os.sep + 'settings.pickle', 'wb') as fid:
            pickle.dump(self.input_applied, fid)

        # store state
        with open(stepdir + os.sep + 'state.dat', 'w') as fid:
            fid.write('{}\n'.format(int(self.has_run)))

    def persistency_load(self):
        """Load state of this step from peristency directory
        """
        print('Persistent load - base version', self.name)
        if self.persistent_directory is None:
            return

        stepdir = self.persistent_directory + os.sep + 'step_' + self.name

        if not os.path.isdir(stepdir):
            return

        # load settings from pickle
        pickle_file = stepdir + os.sep + 'settings.pickle'
        if not os.path.isfile(pickle_file):
            return

        with open(pickle_file, 'rb') as fid:
            # really to input_new ???? makes only sense if has_run=True
            self.input_new = pickle.load(fid)

        # load state
        state = bool(
            open(stepdir + os.sep + 'state.dat', 'r').readline().strip()
        )
        print('has_run  from load:', state)
        if state:
            print('applying next input from persistent storage')
            self.apply_next_input()

    def find_previous_step(self, starting_step, search_name):
        if starting_step.name == search_name:
            return starting_step
        if starting_step.prev_step is None:
            return None
        result = self.find_previous_step(starting_step.prev_step, search_name)
        return result
