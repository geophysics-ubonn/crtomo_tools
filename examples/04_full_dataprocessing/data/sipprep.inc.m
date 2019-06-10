1;
% This is the configuration file for sipprep.m.
% This file is an Octave/Matlab file and will be executed as such when included
% into "sipprep.m". All configurations are done in the struct sip_settings.

% IMPORTANT:
% 1) Store every option/setting in the sip_settings - struct !
% 2) Each section has it's own prefix, for example the radic-section has the
%    prefix "radic_"
% 3) Options to be disabled in the filter-section can be set to NaN


%%%%%%%%%%%%%%%%%%%%%%%
% function collection %
%%%%%%%%%%%%%%%%%%%%%%%
% The collection set here decides which functions will be executed. For the
% moment only the following valid options are available:
% 1) all
% 2) all_no_plots
% 3) first_run
% 4) sec_run
% 5) finish
% Documentation can be found in the CRLab manual.

%sip_settings.collection_name = 'first_run';
sip_settings.collection_name = 'all_no_plots'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Measurement device (device-section) %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This option controls the device from which data will be processed
% Possible values:
% 0: Radic 256c
% 1: Medusa
% 2: Syscal
% 3: Resecs
% 4: .crt files
sip_settings.device = 1;

%%%%%%%%%%%%%%%
%% CRT-Files %%
%%%%%%%%%%%%%%%

% Path to frequency file.
% Note that we expect normal and reciprocal data to have the same frequencies!
sip_settings.crt_frequency_file = '';

% Use only .crt files with a certain prefix (will be used for normal and
% reciprocal directories)
sip_settings.crt_data_file_prefix = '';

% Although we call them .crt files, the data files can have arbitrary file
% endings (will be applied to normal and reciprocal directories)
sip_settings.crt_data_file_ending = 'crt';

% Data dir containing the normal .crt files
sip_settings.crt_datadir_nor = '';

% Data dir containing the reciprocal .crt files
sip_settings.crt_datadir_rec = '';

% Renumber reciprocal electrodes?
% If this settings is not NaN it denotes the maximum number of electrodes.
% Reciprocal quadruples will then be reversed, i.e. electrode 1 will be
% renumbered to the larges electrode number and vice versa.
sip_settings.crt_renumber_rec = NaN;

%%%%%%%%%%%%
%% Resecs %%
%%%%%%%%%%%%

sip_settings.resecs_filename_nor = '';
sip_settings.resecs_filename_rec = '';

%%%%%%%%%%%%%%%%%
%% Radic 256 c %%
%%%%%%%%%%%%%%%%%

% The following options set the data files for normal and
% reciprocal measurements.

% the directory where the data file resides. Can be relative to the location of
% sipprep.inc.m
sip_settings.radic_directory_nor = '../../Data';
% the filename of the data file (without directory structure)
sip_settings.radic_filename_nor = '';

% the directory where the reciprocal data file resides
sip_settings.radic_directory_rec = '../../Data';
% the filename of the data file (without directory structure)
sip_settings.radic_filename_rec = '';

% if the reciprocal data set is in fact a normal data set
% (e.g. in order to compare to measurements), this switch determines
% if the electrode positions are changed when reading in reciprocal
% data
% Default: 0
% Setting to 1 leaves the reciprocal electrodes untouched.
sip_settings.radic_treat_rec_as_nor = 0;

% sometimes, when the software is not restarted or cleared after a measurement, some readings contain more frequencies than others. In those cases you want to remove all frequencies not contained in all readings.
% Example: Filter 125 Hz
%sip_settings.radic_filter_freqs = [125];
sip_settings.radic_filter_freqs = [];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Medusa 2 - (40) Tomograph %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% the directory where the data file reside
sip_settings.medusa_directory = '../../Data';

% the filename of the data file with normalized (on current) potentials for
% each electrode
% (the _einzel.mat file)
sip_settings.medusa_filename_single_pot = 'bnk_raps_20130408_1715_03_einzel.mat';

% Directory where the file containing additional measurement configurations
% is stored.
sip_settings.medusa_directory_scripts = '../../../../Misc/';

% File containing additional measurement configuration to extract from the
% single potential file.
sip_settings.medusa_script = 'configs.inc.m';

% do we want to average the three current injections per measurement?
% sip_settings.medusa_average_swapped_injections = 0
sip_settings.medusa_use_z3_mean = 1;

% if we do not average the three injections per measurement,
% which column to take (1-3)
sip_settings.medusa_use_z3_index = 1;

% Normally we average injections, but sometimes we do not want to do that
% 1: average current injections
% 0: do not avarage current injections
sip_settings.medusa_average_injections = 1;

% Use first (1) or second (2) injection
% This options is only evaluated in case of
% sip_settings.medusa_average_swapped_injections == 1
sip_settings.medusa_use_injection = 1;

%%%%%%%%%%%%
%% Syscal %%
%%%%%%%%%%%%

% The following options set the data files for normal and
% reciprocal measurements.

% the directory where the data file resides. Can be relative to the location of
% sipprep.inc.m
sip_settings.syscal_directory_nor = '../../Data';
% the filename of the data file (without directory structure)
sip_settings.syscal_filename_nor = '';

% sometimes normal and reciprocal measurements are contained in one file. When
% this options is turned on (=1), syscal_filename_nor is assumed to contain
% both normal and reciprocal measurements. When this option is turned off (=0),
% it is assumed that reciprocal data is % present in syscal_filename_rec, or not
%available.
sip_settings.syscal_norrec_in_one_file = 0;

% the directory where the reciprocal data file resides
sip_settings.syscal_directory_rec = '../../Data';
% the filename of the data file (without directory structure)
sip_settings.syscal_filename_rec = '';

% if normal and reciprocal data is stored in separate files, electrode ids
% sometimes need to be renamed
% 1: The same measurement protocoll used for the normal measurements was used
%    for the reciprocal measurements. The physical cable connectors where
%    changed in order to measure reciprocals.
%    The electrode identifier need to be relabeled
% 0: Cable connectors where left unchanged, and different measurement protocols
%    where used. No relabeling necessary.
sip_settings.syscal_relabel_rec_ids = 0;

% Provide assignments of syscal electrode coordinates to electrode numbers:
% SID ID X Z
% Won't be used if = ''
sip_settings.syscal_elec_assignments_file = '';


% There are multiple syscal formats available. The following options determine
% the columns to use for certain values. If unsure, open the data file using
% dlmread in octave and count the columns by hand for each value.

% How many columns should be ignored from the left. This number has to be added
% to the following syscal_column_ options to get the real column number in the
% data file!
% Note: If the data lines start with " Mixed / non conventional ", most of the
% time a value of 4 will remove the zero columns created by dlmread for the
% string.
sip_settings.syscal_ignore_initial_columns = 4;

sip_settings.syscal_column_resistance = 5;
% Denoted by M in head of data file
sip_settings.syscal_column_chargeability = 7;
sip_settings.syscal_column_voltage = 9;
sip_settings.syscal_column_current = 10;

%%%%%%%%%%%%%%%
%% K-Factors %%
%%%%%%%%%%%%%%%

% Calculate K factors for all measurements where a NaN value for K is set.
% 0 = Do not calculate
% 1 = Calculate
sip_settings.K_calculate_numerically = 1;

% The K factor are modelled numerically using CRMod. There, elem.dat and
% elec.dat files need to be provided.
sip_settings.K_elem_file = 'inversions/elem.dat';
sip_settings.K_elec_file = 'inversions/elec.dat';

% 2D or 2.5D mode
% 0: 2.5D
% 1: 2D
sip_settings.K_2D_mode = 1;

% fictitious sink
% if set to a positive number, this number will be used as the node number of
% the sink
sip_settings.K_fictitious_sink = 6467;

% Instead of calculating K factors, a file containing the K factors can be
% supplied. This option is evaluated after 'K_calculate_numerically', thus K
% factors calculated numerically can be overwritten by user supplied ones.  Note
% that the K factor of a reciprocal configuration is equal to that of its
% corresponding normal configuration: K(ABMN) == K(NMBA). Therefore, only the
% factors for all normal measurements need to be supplied.
sip_settings.K_user_supplied = 0;

sip_settings.K_user_file_nor = '';
sip_settings.K_user_file_rec = '';

%%%%%%%%%%%%%%%%%%%%%%%%
%% Correction Factors %%
%%%%%%%%%%%%%%%%%%%%%%%%

% when measuremens are done on a liquid or material with known conductivity,
% correction factors can be calculated that account for the extension of the
% measurement area in the y-direction. If this extension is not infinite, it
% cannot be modelled using the 2D or 2.5D modes of CRMod. Therefore, the
% calculated resistances and resistivities need to be corrected.

% Note: When this option is turn on, the correction factors will be written to
% a file and then the program will be terminated, as further analysis should not
% be necessaey.

% Calculate correction factors: Possible settings:
% 0 : do not calculate corr. factors
% 1 : calculate corr. factors
sip_settings.correction_factors_calc = 0;

% Conductivity of the fill liquid as measured in-situ
% Set in mSiemens/m [mS/m]
sip_settings.correction_factors_cond = -1;

% Load correction factors from file
% If set to NaN, no correction factors will be read in
sip_settings.correction_factors_file_nor = '../../../../Misc/corr_fac_avg_nor.dat';

sip_settings.correction_factors_file_rec = '../../../../Misc/corr_fac_avg_rec.dat';

%%%%%%%%%%%%%%%%%%%%%
%% Filter settings %%
%%%%%%%%%%%%%%%%%%%%%

% NOTE(1): Some filters can be specified frequency wise or globally.
%          In this case the filter variables need to be initilialized with the
%          dimensions equally to the number of frequencies. For example the
%          filter_current_min variable is the initialized by:
%
%          sip_settings.filter_current_min = ones(14,1) .* NaN;
%
%          For 14 frequencies. Replace NaN by a default threshold if only few
%          frequencies get special treatment. Change single frequency with:
%
%          sip_settings.filter_current_min(4) = 24;
%

%%% GLOBAL FILTERS - applied on both normal and reciprocal data  %%%

% Data which lies outside a certain standard deviation of normal-reciprocal
% differences is rejected.
% Note: This filter uses the differences of normal-reciprocal data that was
% computed before any normal/reciprocal filter was applied to the data. Thus the
% magnitude filter does not influence the pha or appmag filters.
sip_settings.filter_norrec_std_mag = NaN;
sip_settings.filter_norrec_std_appmag = NaN;
sip_settings.filter_norrec_std_pha = NaN;

sip_settings.filter_norrec_pha_min = -5;
sip_settings.filter_norrec_pha_max = 5;

sip_settings.filter_norrec_mag_min = NaN;
sip_settings.filter_norrec_mag_max = NaN;
sip_settings.filter_norrec_appmag_min = NaN;
sip_settings.filter_norrec_appmag_max = NaN;

% TODO: sip_settings.filter_norrec_relative_error_mag = NaN;
% TODO: sip_settings.filter_norrec_relative_error_pha = NaN;
% TODO: sip_settings.filter_norrec_relative_error_appmag = NaN;

% current quantile filter
% Delete all data sets whose current is outside the upper and lower limit
% Quantiles are specified in the range [0, 1], 0 meaning 0%, 1 meaning 100%.

% Filter all data whose injectoin current lies below the quantile
sip_settings.filter_current_quantile_lower = NaN;
% Filter all data whose injection current lies above the quantile
sip_settings.filter_current_quantile_upper = NaN;

% Filter K factors >= threshold
sip_settings.filter_Kfactor = 400;

% Only keep data points which retain more than the given percentage of data
% points, after all other filters were applied: Valid values ]0, 100]
sip_settings.filter_incomplete_percentage = 90;

% the percentage above can also be restricted to the frequencies below a
% certain frequency threshold.
sip_settings.filter_incomplete_below_f = 300;

% Filter by number (line number in ascii_data/magnitude_phase/raw_*)
% Provide filename with one number per line
sip_settings.filter_by_id_nor = NaN;
sip_settings.filter_by_id_rec = NaN;

%%% NORMAL %%%

% Per-frequency filters:
% Some filters can be set on a per-frequency base. This is noted in the comments
% of each filter, and can be used by treating the filter as a matlab/octave
% array.
% Example:
% %one threshold for all frequencies
% sip_settings.filter_appmag_min = 100;
% % individual thresholds for all 5 frequencies.
% sip_settings.filter_appmag_min = ones(5,1) * NaN;
% sip_settings.filter_appmag_min(1) = 4;
% sip_settings.filter_appmag_min(2) = 7;
% sip_settings.filter_appmag_min(3) = 10;
% sip_settings.filter_appmag_min(4) = 50;
% sip_settings.filter_appmag_min(5) = 100;

%% Electrode filters
%% Are applied to all frequencies!
%% Note: When determining electrodes to be filtered using the pseudoplots, keep
%%       in mind that the number of the corresponding tick determines the UPPER
%%       dipole id!

% Voltage electrode filters
% Note: The voltage and current electrode filter works on the real electrode
% numbers, NOT the dipole ids!
% Can be a matrix of dimensions: nr_of_filters x 2
% Second entry in each line can be NaN; in that case all voltage measurements
% containing the first electrode will be filtered.
% Example:
%sip_settings.filter_volt_elecs = [40, NaN;
%                                  41, NaN;
%                                  42, NaN;
%                                 [43:49]', [43:49]' .* NaN
%                                  ];
sip_settings.filter_volt_elecs = [13, NaN; 12, NaN];

% Current electrode filters
% Filter all measurements where a certain set of electrodes was used for
% current injection
sip_settings.filter_cur_elecs = [13, NaN; 12, NaN];

% Current and voltage electrode identifiers
sip_settings.filter_current_ids = [NaN];
sip_settings.filter_voltage_ids = [NaN];

% Current filter
% TODO: Description of per-frequency filter
% Example:
%sip_settings.filter_current_min = ones(14,1) .* NaN;
%sip_settings.filter_current_min(1) = ;
%sip_settings.filter_current_min(2) = ;
%sip_settings.filter_current_min(3) = ;
%sip_settings.filter_current_min(4) = ;
%sip_settings.filter_current_min(5) = ;
%sip_settings.filter_current_min(6) = ;
%sip_settings.filter_current_min(7) = ;
%sip_settings.filter_current_min(8) = ;
%sip_settings.filter_current_min(9) = ;
%sip_settings.filter_current_min(10) = ;
%sip_settings.filter_current_min(11) = ;
%sip_settings.filter_current_min(12) = ;
%sip_settings.filter_current_min(13) = ;
%sip_settings.filter_current_min(14) = ;

sip_settings.filter_current_min = NaN;

sip_settings.filter_current_max = NaN;
%sip_settings.filter_current_max = ones(14,1) .* NaN;
%sip_settings.filter_current_max(1) = ;
%sip_settings.filter_current_max(2) = ;
%sip_settings.filter_current_max(3) = ;
%sip_settings.filter_current_max(4) = ;
%sip_settings.filter_current_max(5) = ;
%sip_settings.filter_current_max(6) = ;
%sip_settings.filter_current_max(7) = ;
%sip_settings.filter_current_max(8) = ;
%sip_settings.filter_current_max(9) = ;
%sip_settings.filter_current_max(10) = ;
%sip_settings.filter_current_max(11) = ;
%sip_settings.filter_current_max(12) = ;
%sip_settings.filter_current_max(13) = ;
%sip_settings.filter_current_max(14) = ;

% magnitude filter (linear) (Ohm)
% per-frequency filter enabled (see current filter explanation)
sip_settings.filter_mag_min = 0;
sip_settings.filter_mag_max = NaN;

% magnitude filter (linear): apparent resistivities
% per-frequency filter enabled (see current filter explanation)
sip_settings.filter_appmag_min = 15;
sip_settings.filter_appmag_max = 35;

% magnitude std filter (apparent resisvitities)
sip_settings.filter_appmag_std = NaN;

% phase filter
% per-frequency filter enabled (see current filter explanation)
sip_settings.filter_pha_min = -40;
%sip_settings.filter_pha_min = ones(14,1) .* NaN;
%sip_settings.filter_pha_min(1) = ;
%sip_settings.filter_pha_min(2) = ;
%sip_settings.filter_pha_min(3) = ;
%sip_settings.filter_pha_min(4) = ;
%sip_settings.filter_pha_min(5) = ;
%sip_settings.filter_pha_min(6) = ;
%sip_settings.filter_pha_min(7) = ;
%sip_settings.filter_pha_min(8) = ;
%sip_settings.filter_pha_min(9) = ;
%sip_settings.filter_pha_min(10) = ;
%sip_settings.filter_pha_min(11) = ;
%sip_settings.filter_pha_min(12) = ;
%sip_settings.filter_pha_min(13) = ;
%sip_settings.filter_pha_min(14) = ;

sip_settings.filter_pha_max = 3;
%sip_settings.filter_pha_max = ones(14,1) .* NaN;
%sip_settings.filter_pha_max(1) = ;
%sip_settings.filter_pha_max(2) = ;
%sip_settings.filter_pha_max(3) = ;
%sip_settings.filter_pha_max(4) = ;
%sip_settings.filter_pha_max(5) = ;
%sip_settings.filter_pha_max(6) = ;
%sip_settings.filter_pha_max(7) = ;
%sip_settings.filter_pha_max(8) = ;
%sip_settings.filter_pha_max(9) = ;
%sip_settings.filter_pha_max(10) = ;
%sip_settings.filter_pha_max(11) = ;
%sip_settings.filter_pha_max(12) = ;
%sip_settings.filter_pha_max(13) = ;
%sip_settings.filter_pha_max(14) = ;

% phase std filter
sip_settings.filter_pha_std = NaN;

%%% RECIPROCAL %%%
% Default: Same as normal
% Note: Reciprocal settings can be changed and set in the same way as the
% normal settings can be set!
sip_settings.rec.filter_current_ids = sip_settings.filter_current_ids;
sip_settings.rec.filter_voltage_ids = sip_settings.filter_voltage_ids;
sip_settings.rec.filter_volt_elecs = sip_settings.filter_volt_elecs;
sip_settings.rec.filter_cur_elecs = sip_settings.filter_cur_elecs;
sip_settings.rec.filter_current_min = sip_settings.filter_current_min;
sip_settings.rec.filter_current_max = sip_settings.filter_current_max;
sip_settings.rec.filter_mag_min = sip_settings.filter_mag_min;
sip_settings.rec.filter_mag_max = sip_settings.filter_mag_max;
sip_settings.rec.filter_appmag_min = sip_settings.filter_appmag_min;
sip_settings.rec.filter_appmag_max = sip_settings.filter_appmag_max;
sip_settings.rec.filter_appmag_std = sip_settings.filter_appmag_std;
sip_settings.rec.filter_pha_min = sip_settings.filter_pha_min;
sip_settings.rec.filter_pha_max = sip_settings.filter_pha_max;
sip_settings.rec.filter_pha_std = sip_settings.filter_pha_std;

%%%%%%%%%%%%%%%%%%
%% Error models %%
%%%%%%%%%%%%%%%%%%

sip_settings.directory_errormod = 'errormod';

%%%%%%%%%%%%%%%%%%%
%% Plot settings %%
%%%%%%%%%%%%%%%%%%%

sip_settings.plot_directory = 'spectra';

% If available, plot errorbars in the data (1) or not(0)
sip_settings.plot_errors = 0;

% If available, plot reciprocal data points together with normal data
sip_settings.plot_recind = 1;

% If available, plot Cole-Cole fit
sip_settings.plot_colecole = 1;

% magnitude and phase min/max values
% both in log10
% example: 1e-1, 5e2, etc.
sip_settings.plot_min_mag = NaN;
sip_settings.plot_max_mag = NaN;

sip_settings.plot_min_pha = NaN;
sip_settings.plot_max_pha = NaN;

%%%%%%%%%%%%%%%%%%%%%%%%%
%% Directory structure %%
%%%%%%%%%%%%%%%%%%%%%%%%%

% the root directory of the data structure
sip_settings.directory_root = '.';

% the directory where all plots will be saved (in subdirectories)
sip_settings.directory_plot = 'plots';

% all inspection data should go here
sip_settings.directory_inspection = 'inspection';

% volt.dat style files go to this directory
sip_settings.directory_crtdata = 'crtdata';

% Where to put general histograms
sip_settings.directory_histograms = 'histograms';

%% Inspection of raw data %%

% histogram subdirectory of inspection data
sip_settings.inspec_hist_dir = 'histogram';

%% Raw spectra fits %%

% In which subdirectoryof directory_plot should the raw-spectra be saved
sip_settings.rawspec_dir = 'rawspec';

%% Psuedoplots %%
sip_settings.directory_pseudoplot = 'pseudoplots';
sip_settings.pseudoplot_magdir = 'mag';
sip_settings.pseudoplot_appmagdir = 'appmag';
sip_settings.pseudoplot_phadir = 'pha';

% Both limits need to be set in order to change the limits
% Do disable the limits, use [NaN, NaN];
sip_settings.pseudoplot_phalimits = [NaN, NaN];
% Magnitude is plotted in log10 scale
sip_settings.pseudoplot_maglimits = [NaN, NaN];
sip_settings.pseudoplot_appmaglimits = [NaN, NaN];
