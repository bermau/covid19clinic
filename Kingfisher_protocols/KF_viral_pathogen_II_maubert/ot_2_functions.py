# Define Reagents as objects with their properties
class Reagent():
    def __init__(self,  name, flow_rate_aspirate, flow_rate_dispense, rinse, reagent_reservoir_volume, 
                 delay, num_wells, h_cono, v_fondo, tip_recycling = 'none' , rsup_cono= None):
        self.name = name
        self.flow_rate_aspirate = flow_rate_aspirate
        self.flow_rate_dispense = flow_rate_dispense
        self.rinse = bool(rinse)
        self.reagent_reservoir_volume = reagent_reservoir_volume
        self.delay = delay
        self.num_wells = num_wells
        self.col = 0
        self.vol_well = 0
        self.h_cono = h_cono
        self.v_cono = v_fondo
        self.unused = []
        self.tip_recycling = tip_recycling
        self.vol_well_original = reagent_reservoir_volume / num_wells
        self.rsup_cono = rsup_cono
        self.update()
    
    def update(self):
        pass
        
# Custom functions
def generate_source_table(source):
    """
    Concatenate the wells from the different origin racks
    """
    for rack_number in range(len(source)):
        if rack_number == 0:
            s = source[rack_number].wells()
        else:
            s = s + source[rack_number].wells()
    return s


def calc_height(reagent, cross_section_area, aspirate_volume, min_height=0.5):
    
    global ctx    # ????
    ctx.comment('Remaining volume ' + str(reagent.vol_well) +
                '< needed volume ' + str(aspirate_volume) + '?')
    if reagent.vol_well < aspirate_volume:
        reagent.unused.append(reagent.vol_well)
        ctx.comment('Next column should be picked')
        ctx.comment('Previous to change: ' + str(reagent.col))
        # column selector position; intialize to required number
        reagent.col = reagent.col + 1
        ctx.comment(str('After change: ' + str(reagent.col)))
        reagent.vol_well = reagent.vol_well_original
        ctx.comment('New volume:' + str(reagent.vol_well))
        height = (reagent.vol_well - aspirate_volume - reagent.v_cono) / cross_section_area
                #- reagent.h_cono
        reagent.vol_well = reagent.vol_well - aspirate_volume
        ctx.comment('Remaining volume:' + str(reagent.vol_well))
        if height < min_height:
            height = min_height
        col_change = True
        
    else:
        height = (reagent.vol_well - aspirate_volume - reagent.v_cono) / cross_section_area #- reagent.h_cono
        reagent.vol_well = reagent.vol_well - aspirate_volume
        ctx.comment('Calculated height is ' + str(height))
        if height < min_height:
            height = min_height
        ctx.comment('Used height is ' + str(height))
        col_change = False
        
    return height, col_change
        
##########
# pick up tip and if there is none left, prompt user for a new rack
def pick_up(pip):
    #  nonlocal tip_track
    if not ctx.is_simulating():
        if tip_track['counts'][pip] == tip_track['maxes'][pip]:
            ctx.pause('Replace ' + str(pip.max_volume) + 'µl tipracks before \
            resuming.')
            pip.reset_tipracks()
            tip_track['counts'][pip] = 0
    pip.pick_up_tip()       
    
def find_side(col):
    '''
    Detects if the current column has the magnet at its left or right side
    '''
    if col % 2 == 0:
        side = -1  # left
    else:
        side = 1
    return side
 
#####################################################
## Functions to manage the local order of run
#####################################################

# Par défaut, la fonction wells() ...

# generate the order of the wells walking bottom to top then left to right
# Order is : generate_wells_order(8,12) : [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12,
# 11, 10, 9, 8, ...
# fist position (bottom left is 0)
# return and index 

def generate_wells_order(rows, cols): 
    """
    rows : number of rows
    cols : number of cols
    >>> generate_wells_order(3, 4)
    [2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9]
    """
    
    bottom_wells_list = [(i*rows)-1 for i in range(1, cols+1)]
    my_list =  [ i for N in bottom_wells_list for  i in range(N, N-rows,-1) ]
    return my_list

### Créer un générateur

# Repartition of samples of several racks
# Case of 4  4x6 racks
#  disposition of racks : 
    # rack 1  | rack 2
    # rack 3  | rack 4

def generator_for_4_racks_of_24(racks):
    """up to 96 tubes are loaded in 4 24 tubes racks. 
    The first tube is H1, in rack3, first bottom left, 
    The second tube is G1, in rack3, second bottom left, 
    fith tube is D1, in rack 1, first bottom left...
    
    Returns a well"""
    counter = 0
    
    for rack in racks:
        rack.used_pos = 0
        rack.ordered_wells = generate_wells_order(4, 6)
        
    for rack in [racks[2], racks[0]]*6:
        for i in range(4):
            counter += 1       
            rack.used_pos += 1
            yield rack.wells()[rack.ordered_wells[rack.used_pos-1]]  
 
    for rack in [racks[3], racks[1]]*6:
        for i in range(4):
            counter += 1
            rack.used_pos += 1
            yield rack.wells()[rack.ordered_wells[rack.used_pos-1]]
            
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
def reverse_order_wells(rack, wells_per_columns):
    return [ elt for line in chunks(rack.wells(), wells_per_columns) for elt in line[: : -1]]


# DECK  SUMMARY : 
def short_name(string, nb_cars):
    """Return a max of nb_cars. Or return beginning and end of the 
    string if string is too long
    
    >>> short_name("unnomtroplongpourtenirdanslacase", 25)
    'unnomtropl...irdanslacase'
    """
    length = len(string)
    if length <= nb_cars:
        return string
    else:
        return string[:10] + '...' + string[-(nb_cars -3 -10):]
    
def deck_summary(ctx):
    """Display the deck and its labware"""
    for pos_range  in [[10, 11, 12], [7, 8, 9], [4, 5, 6], [1, 2, 3]]:
        print('| ', end = '')
        for pos in pos_range:
            rack = ctx.deck[pos]            
            if rack:
                try:
                    print("{:<27}".format(str(pos) + ": "
                          + short_name(rack._name,23 )), end=' | ')          
                except:
                    pass  # Trash has no _name.
                    print("{:<27}".format(str(pos) + ": Trash"), end=' | ')
            else:
                print("{:<27}".format(str(pos) + ': ***'), end=' | ')
        print()