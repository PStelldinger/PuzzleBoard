import cv2
import numpy as np
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from sklearn.neighbors import NearestNeighbors
from queue import PriorityQueue

from puzzle_board.cornerdetection import sub_pixel_detection
from puzzle_board.cornerdetection.hessian_detector import HessianDetector, image_regional_max_as_binary_matrix
from puzzle_board.cornerdetection.corner_checker import CornerChecker
from puzzle_board.grid_generator import Grid
from puzzle_board.puzzle_board_decoder import PuzzleBoard


def subpix_pos(img, corners):
    for i in range(corners.shape[0]):
        c=corners[i,:]
        c0=c[0]
        x=int(c0[0])
        y=int(c0[1])
        im=img[max(0,x-12):min(img.shape[0]-1,x+13),max(0,y-12):min(img.shape[1]-1,y+13)]
        if(min(im.shape)<5):
            continue

        H_elems = hessian_matrix(im, sigma=1.0)
        maxima_ridge, minima_ridge = hessian_matrix_eigvals(H_elems)
        fxx=H_elems[0]
        fxy=H_elems[1]
        fyy=H_elems[2]
        S=np.square(fxy)-3*fxx*fyy-np.square(fxx)-np.square(fyy)
        S=S/max(0.00001,np.max(S))*255
        S=S*(S>0)
        S=S.astype(np.uint8)

def detect_puzzleboard(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
    gray = gray/255.

# For low resolution images, there should be less blurring:
#    img2 = cv2.filter2D(src=gray, ddepth=-1, kernel=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    img2 = cv2.GaussianBlur(gray, (3, 3), 1)

    h_d = HessianDetector(img2)

    profile, mS = h_d.detect_corners(img2, k=1.0)
    
    corner_checker=CornerChecker()
    mS_corner = corner_checker.filter_corners(image=img2)

    mS_corner_binary = np.where(mS_corner > 0, 1, 0)

    mS = mS * mS_corner_binary
    mS = mS / np.nanmax(mS + 0.00000001)
    mS[mS < 0.03] = 0.0
    mS = image_regional_max_as_binary_matrix(mS) * mS

    dot_row, dot_col = np.where(mS > 0)

    dot = np.column_stack((dot_row, dot_col))

    dot = dot[dot[:, 0] > 3]
    dot = dot[dot[:, 1] > 3]
    dot = dot[dot[:, 0] <= img.shape[0] - 3]
    dot = dot[dot[:, 1] <= img.shape[1] - 3]

    sub_dot = (sub_pixel_detection.get_subpixel_positions(profile, mS, dot))

    root = np.sqrt((h_d.f_xx - h_d.f_yy)**2 + 4 * h_d.f_xy**2)
    first_eigenvector_x  = h_d.f_xx - h_d.f_yy + root
    second_eigenvector_x = h_d.f_xx - h_d.f_yy - root
    both_eigenvectors_y  = -2 * h_d.f_xy

    # Berechne X-Wert und gemeinsamen Y-Wert des ersten (positiven) und des zweiten (negativen) Eigenvektors
    ev1_x_at_max = np.fromiter(( first_eigenvector_x[dot[idx, 0], dot[idx, 1]] for idx in range(len(dot))),float)
    ev2_x_at_max = np.fromiter((second_eigenvector_x[dot[idx, 0], dot[idx, 1]] for idx in range(len(dot))),float)
    ev_y_at_max  = np.fromiter(( both_eigenvectors_y[dot[idx, 0], dot[idx, 1]] for idx in range(len(dot))),float)

    # Nutze Eigenvektor mit betragsmäßig größerem X-Wert (numerisch stabiler)
    # Wenn dies der 2. Eigenvektor ist, drehe diesen um 90 Grad ([X,Y]->[Y,-X])
    evx_at_max = ev1_x_at_max.copy()
    evy_at_max = ev_y_at_max.copy()
    evx_at_max[np.abs(ev2_x_at_max) > np.abs(ev1_x_at_max)] =  ev_y_at_max[np.abs(ev2_x_at_max) > np.abs(ev1_x_at_max)]
    evy_at_max[np.abs(ev2_x_at_max) > np.abs(ev1_x_at_max)] = -ev2_x_at_max[np.abs(ev2_x_at_max) > np.abs(ev1_x_at_max)]
    eigenvectors_at_max = np.concatenate((evx_at_max, evy_at_max)).reshape((-1, 2), order='F')
    first_eigenvector_at_max = eigenvectors_at_max[:] / (1e-30+np.linalg.norm(eigenvectors_at_max, axis=1).reshape(-1,1))

    NUMBER_WANTED_NEIGHBORS = 10
    if sub_dot.shape[0] > NUMBER_WANTED_NEIGHBORS:
        nbrs = NearestNeighbors(n_neighbors=NUMBER_WANTED_NEIGHBORS+1, algorithm='auto').fit(sub_dot)
        distances, idx_neighbors = nbrs.kneighbors(sub_dot)
        distances = distances[:,1:]
        idx_neighbors = idx_neighbors[:,1:]


        orientation_diff_angle = np.abs(np.sum((first_eigenvector_at_max.reshape(-1,1,2)*first_eigenvector_at_max[idx_neighbors,:]),axis=2))
        thr1 = 0.38268  # 0.38268 => 67.5°
        thr2 = 0.92388  # 0.92388 => 22.5°
        thr3 = 0.98481  # 0.92388 => 10.0°

        neighb_mask = orientation_diff_angle < thr1   
        diagonal_mask = orientation_diff_angle > thr2 
    
        has_neighbor= np.full((dot.shape[0],5), False)
        neighbor_idx= np.full((dot.shape[0],5), 0)

        # Find the nearest direct neighbor
        far_enough_mask = np.cumprod(1 - neighb_mask, axis=1)
        nearest_neighbor = np.sum(far_enough_mask, axis=1)
        far_enough_mask = 1 - far_enough_mask
        has_neighbor[:,0] = (nearest_neighbor<10)
        nearest_neighbor = nearest_neighbor * has_neighbor[:,0]
        neighbor_idx[:,0] = idx_neighbors[np.arange(len(idx_neighbors)), nearest_neighbor]
        
        # Achtung! Vorher müssen noch die Kollinearen Versionen des nächsten Nachbarn aussortiert werden!
        dir1 = sub_dot[neighbor_idx[:,0]] - sub_dot
        len1 = np.linalg.norm(dir1,axis=1).reshape(-1,1)
        dir1 = dir1 / len1
        dir2 = sub_dot[idx_neighbors,:]-sub_dot.reshape(-1,1,2)
        len2 = np.expand_dims(np.linalg.norm(dir2,axis=2),axis=2)
        dir2 = dir2 / len2
        direction_diff_angle = np.sum(dir1.reshape(-1,1,2)*dir2,axis=2)
        not_collinear_mask = np.abs(direction_diff_angle) < thr3
        opposite_mask = direction_diff_angle < -thr3
        length_mask = (np.maximum(len1.reshape(-1,1,1) / len2, len2 / len1.reshape(-1,1,1)) <= 1.5).reshape(-1,10)

        nearest_diagonal = np.sum(np.cumprod(1 - diagonal_mask * not_collinear_mask * far_enough_mask, axis=1), axis=1)
        nearest_opposite = np.sum(np.cumprod(1 - neighb_mask * opposite_mask * length_mask, axis=1), axis=1)
        
        nearest_orthogonal = 0 * nearest_diagonal + NUMBER_WANTED_NEIGHBORS  # Inizialization: Everything is 10 (no orthogonal found)
        for i in range(len(dot)):
            if(not has_neighbor[i,0]): continue
            nb1_nr = nearest_neighbor[i]
            nb1 = idx_neighbors[i,nb1_nr]   # index of nearest neighbor with appropriate orientation
            has_neighbor[i,0] = True
            neighbor_idx[i,0] = nb1
            nb1o_nr = nearest_opposite[i]
            nb1o=nb1
            if(nb1o_nr<NUMBER_WANTED_NEIGHBORS):
                nb1o = idx_neighbors[i,nb1o_nr]   # index of nearest opposite neighbor with appropriate orientation
                has_neighbor[i,2] = True
                neighbor_idx[i,2] = nb1o

            nb2_nr=nearest_diagonal[i]          
            if(nb2_nr<NUMBER_WANTED_NEIGHBORS):
                nb2 = idx_neighbors[i,nb2_nr]   # index of nearest diagonal neighbor (with wrong orientation)
                has_neighbor[i,4] = True
                neighbor_idx[i,4] = nb2
            if(nb1_nr<NUMBER_WANTED_NEIGHBORS) and (nb2_nr<NUMBER_WANTED_NEIGHBORS):
                v1 = sub_dot[nb1,:] - sub_dot[i,:]
                nb1_mirrored1 = v1 - 2 * (np.dot(v1, first_eigenvector_at_max[i,:]) * first_eigenvector_at_max[i,:])
                nb1_mirrored1 = nb1_mirrored1 / np.linalg.norm(nb1_mirrored1)
                d2=sub_dot[nb2,:]-sub_dot[nb1,:]
                if(nb1o_nr<NUMBER_WANTED_NEIGHBORS):
                    d3=sub_dot[nb2,:]-sub_dot[nb1o,:]
                else:
                    d3=sub_dot[nb2,:]+sub_dot[nb1,:]-2*sub_dot[i,:]
                nd2=d2/np.linalg.norm(d2)
                nd3=d3/np.linalg.norm(d3)
                guess = d2 + sub_dot[i,:]
                if np.abs(np.dot(nb1_mirrored1,nd2)) < np.abs(np.dot(nb1_mirrored1,nd3)):
                    guess = d3 + sub_dot[i,:]

                vecs = sub_dot[idx_neighbors[i,:]]-sub_dot[i,:]   
                lens = np.linalg.norm(vecs,axis=1).reshape([-1,1])
                nvecs = vecs / lens
                
                pos = np.argmax(neighb_mask[i,:]/(1+np.linalg.norm(sub_dot[idx_neighbors[i,:]]-guess,axis=1))) # Nächster zulässiger Punkt an guess
                nearest_orthogonal = idx_neighbors[i, pos]
                if nearest_orthogonal==nb1 or nearest_orthogonal==nb1o:
                    continue
                has_neighbor[i,1] = True
                neighbor_idx[i,1] = nearest_orthogonal

                opp_orth_mask = (np.sum(nvecs*nvecs[pos,:].T,axis=1) < -thr3)
                length_mask2 = (np.maximum(lens / lens[pos], lens[pos] / lens) <= 1.5)
                nearest_opp_orth = np.sum(np.cumprod(1 - neighb_mask[i,:] * opp_orth_mask * length_mask2))
                if(nearest_opp_orth<NUMBER_WANTED_NEIGHBORS):
                    nb2o = idx_neighbors[i,nearest_opp_orth]   # index of nearest opposite orth neighbor with appropriate orientation
                    has_neighbor[i,3] = True
                    neighbor_idx[i,3] = nb2o
                
                # Fix order of the neighbors to clockwise order:
                pt = sub_dot[i,:]
                pt0 = sub_dot[neighbor_idx[i,0],:] - pt
                pt1 = sub_dot[neighbor_idx[i,1],:] - pt
                if np.sum(np.array([pt0[1],-pt0[0]])*pt1)<0:
                    neighbor_idx[i,1], neighbor_idx[i,3] = neighbor_idx[i,3], neighbor_idx[i,1]
                    has_neighbor[i,1], has_neighbor[i,3] = has_neighbor[i,3], has_neighbor[i,1]

        nodes = []
        Grid.reset()
        for i in range(len(sub_dot)):
            nodes.append(Grid(i))
        for i in range(len(sub_dot)):
            if has_neighbor[i,0]:
                nodes[i].set_left(nodes[neighbor_idx[i,0]])
            if has_neighbor[i,1]:
                nodes[i].set_top(nodes[neighbor_idx[i,1]])
            if has_neighbor[i,2]:
                nodes[i].set_right(nodes[neighbor_idx[i,2]])
            if has_neighbor[i,3]:
                nodes[i].set_bottom(nodes[neighbor_idx[i,3]])

        pq = PriorityQueue()
        for i in range(len(sub_dot)):
            for k in range(4):
                if has_neighbor[i,k]:
                    j = neighbor_idx[i,k]
                    pq.put((1.0/(nodes[i].nr_neighbors*mS[dot[i,0],dot[i,1]] + nodes[j].nr_neighbors*mS[dot[j,0],dot[j,1]]), (i,j)))
        while pq.qsize()>0:
            edge0 = pq.get()
            edge = edge0[1]
            i = edge[0]
            j = edge[1]
            nodes[i].connect(nodes[j])
                
        first_root = Grid.first_root
        root = first_root
        col_nr = 0
        root_nr = 0
        nrBoards = 0
        
        point_ids = []
        point_coords = []
        
        while not (root is None):
            node = root
            dim = root.dimensions
            if (root.size >= 12) and (min(dim[0]+dim[2]+1,dim[1]+dim[3]+1) >= 4):
                nrBoards = nrBoards + 1
                col_nr = (col_nr + 1) % 6
                board = PuzzleBoard(node, sub_dot, img2)
                
                for y in range(board.hvalid.shape[0]):
                    for x in range(board.hvalid.shape[1]):
                        if board.hvalid[y,x]:
                            point_ids.append(np.array(board.positions[:,y,x]))
                            point_coords.append(np.array(board.sub_dot[y,x,:]))
                    
            root = root.next_root
            root_nr = root_nr + 1
            if root == first_root:
                break


    return point_ids, point_coords
